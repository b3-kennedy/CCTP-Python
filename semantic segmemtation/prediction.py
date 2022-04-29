

import torch


def predict(img,model,preprocess,postprocess,device):
    model.eval()
    img = preprocess(img)
    x = torch.from_numpy(img).to(device)
    with torch.no_grad():
        out,x = model(x)

    out_softmax = torch.softmax(out, dim=1)
    result = postprocess(out_softmax)

    return result



