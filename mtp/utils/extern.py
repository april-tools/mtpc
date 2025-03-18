import torch


@torch.compile(fullgraph=True, mode='reduce-overhead')
def log1mexp(x):
    # Source: https://github.com/UCLA-StarAI/SIMPLE/blob/main/v2/simple.py#L32
    # Source: https://github.com/wouterkool/estimating-gradients-without-replacement/blob/9d8bf8b/bernoulli/gumbel.py#L7-L11
    
    # Computes log(1-exp(-|x|))
    # See https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    x = -x.abs()
    x = torch.where(
        x > -0.6931471805599453094,
        torch.log(-torch.expm1(x)),
        torch.log1p(-torch.exp(x)),
    )
    return x
