import torch
import torch.nn.functional as F

from sensa.loss.base import BaseLoss
from sensa.loss.registry import register_loss


@register_loss("DinoLoss")
class DinoLoss(BaseLoss):
    def __init__(
        self,
        dim: int,
        epochs: int,
        student_temperature: float = 0.1,
        teacher_temperature: float = 0.04,
        warmup_teacher_temperature: float = 0.04,
        warmup_teacher_temperature_epochs: int = 30,
        center_momentum: float = 0.9,
    ):
        super().__init__()
        self.student_temperature = student_temperature
        self.teacher_temperature = teacher_temperature
        self.warmup_teacher_temperature = warmup_teacher_temperature
        self.warmup_teacher_temperature_epochs = warmup_teacher_temperature_epochs
        self.center_momentum = center_momentum
        # centers
        self.register_buffer("center", torch.zeros(1, dim))
        # linear scheduler
        from sensa.trainer.scheduler import fn_linear

        self.scheduler = fn_linear

    def forward(self, student_outputs: torch.Tensor, teacher_outputs: torch.Tensor, epoch: int) -> torch.Tensor:
        bsize = teacher_outputs.shape[0] / 2
        num_local_crops = int(student_outputs.shape[0] / bsize) - 2
        souts = (student_outputs / self.student_temperature).chunk(2 + num_local_crops)

        # teacher centering and sharpening
        teacher_temperature = self.scheduler(
            start=self.warmup_teacher_temperature,
            end=self.teacher_temperature,
            iteration=epoch,
            iterations=self.warmup_teacher_temperature_epochs,
        )
        teacher_outputs = F.softmax((teacher_outputs - self.center) / teacher_temperature, dim=-1)

        loss: torch.Tensor | float = 0.0
        loss_terms: int = 0
        for i, to in enumerate(teacher_outputs.detach().chunk(2)):
            for j, so in enumerate(souts):
                if i == j:
                    continue
                loss = loss + torch.sum(-to * F.log_softmax(so, dim=-1), dim=-1).mean()
                loss_terms += 1

        self.update_center(teacher_outputs)
        return loss / loss_terms

    @torch.no_grad()
    def update_center(self, teacher_outputs: torch.Tensor) -> None:
        centers = torch.sum(teacher_outputs, dim=0, keepdim=True)
        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and (torch.distributed.get_world_size() > 1)
        ):
            torch.distributed.all_reduce(centers, op=torch.distributed.ReduceOp.AVG)

        centers = centers / len(teacher_outputs)
        self.center = self.center * self.center_momentum + centers * (1 - self.center_momentum)
