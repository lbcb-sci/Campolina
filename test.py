import torch

from campolina.loss.loss import ConsecPenalty

loss = ConsecPenalty()

# predicting consecutive positives should be penalized
predictions = torch.tensor([1, 2, 3, 4, -1, -2, -3, 1, 2, 3], dtype=torch.float32,requires_grad=True).unsqueeze(0)
x = loss(predictions)
print(x)

# predicting sparse positives should be less penalized
predictions = torch.tensor([1, 0, 1, 0, -1, 1, 0, 1, 0, 1], dtype=torch.float32, requires_grad=True).unsqueeze(0)
x = loss(predictions)
print(x)

# this penalty probably not better than the current consec loss tho 