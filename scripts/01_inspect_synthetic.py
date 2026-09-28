from safeplus.data.synthetic import generate_synthetic

data = generate_synthetic(
    n=10,
    length=24,
    features=8,
    seed=0
)

print("x shape:", data["x"].shape)
print("lengths shape:", data["lengths"].shape)

print("\n前10个用户：")

for i in range(10):
    print(
        f"user={data['entity_id'][i]:2d} | "
        f"detected={data['detected'][i]} | "
        f"tf={data['t_onset'][i]:2d} | "
        f"td={data['t_detect'][i]:2d}"
    )
print("\nInspect user 5:")
user = 5

tf = data["t_onset"][user]
td = data["t_detect"][user]

for t in range(24):
    feature_mean = data["x"][user, t].mean()

    mark = ""

    if t == tf:
        mark += "  <-- tf"

    if data["detected"][user] == 1 and t == td:
        mark += "  <-- td"

    print(
        f"t={t:2d} | "
        f"feature_mean={feature_mean: .3f}"
        f"{mark}"
    )