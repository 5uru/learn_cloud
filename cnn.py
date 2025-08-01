from tinygrad import Tensor, nn
from tinygrad.nn.datasets import mnist
from tinygrad import TinyJit
from tinygrad.nn.state import safe_save, safe_load, get_state_dict, load_state_dict
import trackio as wandb
from huggingface_hub import HfApi
import os
import time

class Model:
    def __init__(self):
        self.l1 = nn.Conv2d(1, 32, kernel_size=(3,3))
        self.l2 = nn.Conv2d(32, 64, kernel_size=(3,3))
        self.l3 = nn.Linear(1600, 10)

    def __call__(self, x:Tensor) -> Tensor:
        x = self.l1(x).relu().max_pool2d((2,2))
        x = self.l2(x).relu().max_pool2d((2,2))
        return self.l3(x.flatten(1).dropout(0.5))


X_train, Y_train, X_test, Y_test = mnist()

model = Model()
optim = nn.optim.Adam(nn.state.get_parameters(model))
batch_size = 128
def step():
    Tensor.training = True
    samples = Tensor.randint(batch_size, high=X_train.shape[0])
    X, Y = X_train[samples], Y_train[samples]
    optim.zero_grad()
    loss = model(X).sparse_categorical_crossentropy(Y).backward()
    optim.step()
    return loss

jit_step = TinyJit(step)
steps = 7000

# Initialize best accuracy tracking
best_acc = 0.0
best_step = -1
start_time = time.time()
# randomn project name
id_ = round(start_time % 10000)
print(f"Project ID: {id_}")
# Initialize wandb once at the beginning
wandb.init(project=f"CNN_{id_}", space_id="CNN", config={
        "steps": steps,
        "learning_rate": 0.001,
        "batch_size": 128,
})

print(f"Starting training for {steps} steps...")

for step_num in range(steps):
    loss = jit_step()
    # Evaluate model periodically
    if step_num % 100 == 0 or step_num == steps-1:
        # Log loss to wandb every step for real-time monitoring
        wandb.log({"train_loss": loss.item()}, step=step_num)
        Tensor.training = False
        acc = (model(X_test).argmax(axis=1) == Y_test).mean().item()

        # Simple print for progress tracking
        print(f"step {step_num:4d}, loss {loss.item():.4f}, acc {acc*100:.2f}%")

        # Log accuracy to wandb
        wandb.log({"test_acc": acc}, step=step_num)

        # Track best model
        if acc > best_acc:
            best_acc = acc
            best_step = step_num
            print(f"★ New best accuracy: {best_acc*100:.2f}% at step {best_step}")
            best_state_dict = get_state_dict(model)
            safe_save(best_state_dict, "best_model.safetensors")
            wandb.log({"best_acc": best_acc, "best_step": best_step})

# Finish wandb session at the end
wandb.finish()

# Final results
print(f"Training completed. Best accuracy: {best_acc*100:.2f}% at step {best_step}")

# Save final model
state_dict = get_state_dict(model)
safe_save(state_dict, "model.safetensors")
print(f"Token exists: {os.getenv('HF_TOKEN') is not None}")
api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_file(
        path_or_fileobj="model.safetensors",
        path_in_repo="model.safetensors",
        repo_id="jonathansuru/cnn",
        repo_type="model",
)

# Upload best model if different from final model
if best_step != steps - 1:
    api.upload_file(
            path_or_fileobj="best_model.safetensors",
            path_in_repo="best_model.safetensors",
            repo_id="jonathansuru/cnn",
            repo_type="model",
    )