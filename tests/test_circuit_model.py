from mtp.models.circuits import CircuitModel


def build_circuit(vocab_size: int, n_token: int, n_component: int, kind: str) -> CircuitModel:
    return CircuitModel(
        vocab_size,
        n_token,
        n_component,
        kind=kind
    )


def fully_factorized(vocab_size: int, n_token: int, n_component: int) -> CircuitModel:
    return build_circuit(vocab_size, n_token, n_component, kind='fully_factorized')


def cp(vocab_size: int, n_token: int, n_component: int) -> CircuitModel:
    return build_circuit(vocab_size, n_token, n_component, kind='cp')


def hmm(vocab_size: int, n_token: int, n_component: int) -> CircuitModel:
    return build_circuit(vocab_size, n_token, n_component, kind='hmm')
