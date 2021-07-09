import torch


class Architect():
    """
    Architect handle optimizer algorithm.
    Update parameters alpha (first step of pc-darts algorithm)
    """
    def __init__(self, model, args):
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
                                          lr=args.arch_lr, betas=(args.beta_grad, args.beta_square),
                                          weight_decay=args.arch_weight_decay)  # betas = (0.9, 0.999)

    def step(self, input_valid, target_valid):
        # 1-step unrolled validation step not considered in the current implementation
        self.optimizer.zero_grad()
        self._backward_step(input_valid, target_valid)
        self.optimizer.step()

    def _backward_step(self, input_valid, target_valid):
        loss = self.model._loss(input_valid, target_valid)
        loss.backward()
