import pytorch_lightning as pl

class CheckBatchGradient(pl.Callback):
    
    def on_train_start(self, trainer, model):
        n = 0

        example_input = model.example_input_array.to(model.device)
        example_input.requires_grad = True

        model.zero_grad()
        output = model(example_input)
        output[n].abs().sum().backward()
        
        zero_grad_inds = list(range(example_input.size(0)))
        zero_grad_inds.pop(n)
        
        if example_input.grad[zero_grad_inds].abs().sum().item() > 0:
            raise RuntimeError("Your model mixes data across the batch dimension!")

def adjust_max_eval(initial_value, current_epoch, milestones=[5, 10, 15], gamma=0.5, threshold=16):
    # Decrease by step_size at each milestone
    if current_epoch in milestones:
        new_value = int(gamma*initial_value)
        if new_value > threshold:
            return new_value
        else:
            return 0
    else:
        return initial_value
    
class VariableAdjustmentCallback(pl.Callback):
    def __init__(self, variable_name, max_epochs, milestones_ratio, adjustment_function=adjust_max_eval, gamma=0.5, threshold=16):
        self.max_epochs = max_epochs
        self.milestones_ratio = milestones_ratio
        self.milestones = [int(max_epochs * ratio) for ratio in milestones_ratio]
        self.variable_name = variable_name
        self.adjustment_function = adjustment_function
        self.gamma = gamma
        self.threshold = threshold

    def on_train_epoch_end(self, trainer, pl_module):
        # Get the current epoch
        current_epoch = trainer.current_epoch
        # Adjust the variable
        previous_value = getattr(pl_module, self.variable_name)
        new_value = self.adjustment_function(previous_value, current_epoch, self.milestones, self.gamma, self.threshold)
        if previous_value != new_value:
            setattr(pl_module, self.variable_name, new_value)
            print(f"Updated {self.variable_name} to {new_value}")



