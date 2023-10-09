import torch
import torch.nn as nn
from distillation.baseDistiller import BaseDistiller
from distillation.utils import Accuracy, AverageMeter, Hook
from tqdm import tqdm
from torch.cuda.amp import autocast
import os


class HintonDistiller(BaseDistiller):

    def __init__(self, alpha, teacher, dataloader, task, studentLayer=-2, teacherLayer=-2):
        super(HintonDistiller, self).__init__()

        self.alpha = alpha
        self.studentLayer = studentLayer
        self.teacherLayer = teacherLayer

        # Register hooks
        self.studentHook = Hook()
        self.teacherHook = Hook()

        self.dataloader = dataloader
        device = next(teacher.parameters()).device

        if os.path.exists(task+"_teacherLogits.pt"):
            ckpt = torch.load(task+"_teacherLogits.pt", map_location="cpu")
            self.tLogits, self.tAct = ckpt['logits'], ckpt['activations']

        else:

            teacher.eval()
            teacher.half()

            if not self.teacherHook.hooked():
                self._setHook(self.teacherHook, teacher, self.teacherLayer)

            # Calculate teacher logits
            self.tLogits = []
            self.tAct = []
            with torch.no_grad():
                for _, data in tqdm(enumerate(dataloader), total=len(dataloader), desc="Calculating teacher logits"):
                    if isinstance(data, dict):
                        target = data.pop('labels').to(device)
                        tdata = {k: v.to(device) for k, v in data.items()}

                    elif len(data) == 3:
                        data, mask, target = data[0].to(device), data[1].to(device), data[2].to(device)
                        sdata = data
                        tdata = {'input_ids': data, 'attention_mask': mask}

                    elif len(data) == 2:
                        data, target = data[0].to(device), data[1].to(device)
                        sdata = data
                        tdata = {'input_ids': data}

                    else:
                        raise ValueError("Dataloader must return a tuple of (data, target) or (data, label, target).")
                    self.tLogits.append(teacher(**tdata).logits.detach().clone().cpu())

                    # Retrieve activations from distillation layer of both models
                    self.tAct.append(self.teacherHook.val().detach().clone().cpu())
            torch.save({'logits': self.tLogits, 'activations': self.tAct}, task+'_teacherLogits.pt')


    def train_step(self, student, optimizer, scheduler, scaler, objective, distillObjective, OneHot=False):
        """
        Train student model to the teacher model for one epoch with Hinton KD.

        :return: dict, named metrics for logging.
        """
        student.train()

        # Attach
        if not self.studentHook.hooked():
            self._setHook(self.studentHook, student, self.studentLayer)

        device = next(student.parameters()).device
        accuracy = Accuracy(OH=OneHot)
        lossMeter = AverageMeter()
        accMeter = AverageMeter()

        pbar = tqdm(enumerate(self.dataloader), total=len(self.dataloader), desc="Distilling")
        for idx, data in pbar:
            if isinstance(data, dict):
                target = data.pop('labels').to(device)
                tdata = {k: v.to(device) for k, v in data.items()}

            elif len(data) == 3:
                data, mask, target = data[0].to(device), data[1].to(device), data[2].to(device)
                sdata = data
                tdata = {'input_ids': data, 'attention_mask': mask}

            elif len(data) == 2:
                data, target = data[0].to(device), data[1].to(device)
                sdata = data
                tdata = {'input_ids': data}

            else:
                raise ValueError("Dataloader must return a tuple of (data, target) or (data, label, target).")

            # Calculate student logits
            tLogits = self.tLogits[idx]

            # Retrieve activations from distillation layer of both models
            tAct = self.tAct[idx].to(device)

            # Calculate loss
            with autocast():
                sLogits = student(**tdata).logits
                sAct = self.studentHook.val()
                # print(sAct.shape, tAct.shape)
                soft_loss = (1 - self.alpha) * distillObjective(
                    nn.functional.log_softmax(sAct, dim=1),
                    nn.functional.softmax(tAct, dim=1),
                )
                hard_loss = self.alpha * objective(
                    nn.functional.log_softmax(sLogits, dim=1),
                    target,
                )
                batchLoss = soft_loss + hard_loss

            # Update student weights
            scale = scaler.get_scale()
            scaler.scale(batchLoss).backward()
            scaler.step(optimizer)
            scaler.update()
            skip_lr_sched = (scale > scaler.get_scale())
            if not skip_lr_sched:
                scheduler.step()
            optimizer.zero_grad()
            # batchLoss.backward()
            # optimizer.step()

            # Save metrics
            batchLoss = batchLoss.item()
            acc = accuracy(nn.functional.softmax(sLogits, dim=1), target)
            lossMeter.update(batchLoss, n=len(data))
            accMeter.update(acc, n=len(data))
            pbar.set_description(f'Distilling (h={hard_loss:.4f}, s={soft_loss:.4f}, a={accMeter.avg:.2%}, lr={scheduler.get_last_lr()[0]:.2e})')

        return {'Train/Loss': lossMeter.avg,
                'Train/Accuracy': accMeter.avg}
