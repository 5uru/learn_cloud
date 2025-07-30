from tinygrad import Tensor, nn
from tinygrad.nn.datasets import mnist
from tinygrad import TinyJit
from tinygrad.nn.state import safe_save, safe_load, get_state_dict, load_state_dict
import trackio as wandb
from huggingface_hub import HfApi
import os

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
print(X_train.shape, X_train.dtype, Y_train.shape, Y_train.dtype)
# (60000, 1, 28, 28) dtypes.uchar (60000,) dtypes.uchar

model = Model()
acc = (model(X_test).argmax(axis=1) == Y_test).mean()

optim = nn.optim.Adam(nn.state.get_parameters(model))
batch_size = 128
def step():
    Tensor.training = True  # makes dropout work
    samples = Tensor.randint(batch_size, high=X_train.shape[0])
    X, Y = X_train[samples], Y_train[samples]
    optim.zero_grad()
    loss = model(X).sparse_categorical_crossentropy(Y).backward()
    optim.step()
    return loss


jit_step = TinyJit(step)
steps = 1
wandb.init(project="CNN", space_id="jonathansuru/CNN", config={
        "steps": steps,
        "learning_rate": 0.001,
        "batch_size": 128,

})
for step_num in range(steps):

    loss = jit_step()
    wandb.log({"train_loss": loss.item()}, step=step_num)
    if step_num%100 == 0:
        Tensor.training = False
        acc = (model(X_test).argmax(axis=1) == Y_test).mean().item()
        print(f"step {step_num:4d}, loss {loss.item():.2f}, acc {acc*100.:.2f}%")
        wandb.log({"test_acc": acc}, step=step_num)

wandb.finish()

# Save model
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

