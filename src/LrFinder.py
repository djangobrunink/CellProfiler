from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm


class LRFinder:
    def __init__(
            self,
            start_lr: float = 1e-7,
            end_lr: float = 10,
            num_it: int = 200,
            beta: float = 0.9
    ):
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.num_it = num_it
        self.beta = beta

    def find(
            self,
            model,
            optimizer,
            data_loader,
    ):
        tmp_folder = Path("./.tmp")
        weight_file = tmp_folder / "weights"

        tmp_folder.mkdir(exist_ok=True)
        torch.save(model.state_dict(), weight_file)

        lr_scheduler = LRFinderSchedular(optimizer, self.start_lr, self.end_lr, self.num_it)

        lrs = []
        loss_measurements = []
        i = 0

        for inputs in tqdm(data_loader, total=self.num_it):
            loss_dict = model(inputs)
            losses = sum(loss_dict.values())

            if torch.isnan(losses):
                break

            lrs.append(optimizer.state_dict()["param_groups"][-1]["lr"])
            loss_measurements.append(float(losses.detach().cpu()))

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            i += 1
            if i >= self.num_it or float(losses) > 3 * loss_measurements[0]:
                break

            lr_scheduler.step()

        model.load_state_dict(torch.load(weight_file))
        weight_file.unlink()

        lrs = np.array(lrs).squeeze()
        loss_measurements = np.array(loss_measurements).squeeze()

        return SuggestedLRs(lrs[:-1], loss_measurements[:-1], self.beta)


class LRFinderSchedular(LambdaLR):
    def __init__(self,
                 optimizer,
                 start_lr: float = 1e-7,
                 end_lr: float = 10,
                 num_it: int = 100,
                 ) -> None:
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.num_it = num_it
        self.scheduled_lrs = np.logspace(np.log10(start_lr), np.log10(end_lr), num_it)

        for param_group in optimizer.param_groups:
            param_group["lr"] = 1

        super(LRFinderSchedular, self).__init__(optimizer, self._lr_lambda_, last_epoch=-1)

    def _lr_lambda_(self, epoch: int):
        return self.scheduled_lrs[epoch]


class SuggestedLRs:
    def __init__(
            self,
            lrs,
            losses,
            beta: float = 0.9
    ) -> None:
        lrs = np.array(lrs)
        losses = np.array(losses)
        assert len(lrs.shape) == 1
        assert lrs.shape == losses.shape
        assert 0 < beta < 1

        self.lrs = lrs
        self.losses = losses
        self.beta = beta

    @property
    def smoothed_losses(self):
        return self._moving_average(self.losses)

    @property
    def gradients(self):
        smoothed_losses = self.smoothed_losses
        return (smoothed_losses[1:] - smoothed_losses[:-1]) / (np.log10(self.lrs[1:]) - np.log10(self.lrs[:-1]))

    @property
    def lr_steep(self):
        return float(self.lrs[np.argmin(self.gradients)])

    @property
    def lr_min(self):
        return float(self.lrs[np.argmin(self.smoothed_losses)] * (1 - self.beta))

    def _moving_average(self, array):
        res = [array[0]]
        for v in array[1:]:
            res.append(self.beta * res[-1] + (1 - self.beta) * v)

        return np.array(res)

    def plot(self):
        plt.plot(self.lrs, self.smoothed_losses)
        plt.xlabel("lr")
        plt.ylabel("losses")
        plt.xscale("log")
        plt.grid()
        plt.show()

    def __repr__(self):
        return f"SuggestedLRs(\n{self.lrs},\n{self.losses},\n)"
