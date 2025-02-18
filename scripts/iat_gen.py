import numpy as np


def genIAT(lambda_rate: float, output: str):
    iats = np.random.exponential(1 / lambda_rate, size=1000)
    iats = np.round(iats)
    unique, counts = np.unique(iats, return_counts=True)
    freq = np.asarray((unique, counts / len(iats) * 100)).T
    if output:
        freq[:, 0] = np.round(freq[:, 0], 1)
        freq[:, 1] = np.round(freq[:, 1], 4)
        np.savetxt(
            output,
            freq,
            delimiter=",",
            header="IAT (sec),prob (%)",
            fmt=("%.1f", "%.4f"),
        )
    return freq


# Generate IAT distributions with different rates (# jobs / hour).
for rate in [1, 5]:
    freq = genIAT(lambda_rate=rate / 3600, output=f"iat{rate}.csv")
