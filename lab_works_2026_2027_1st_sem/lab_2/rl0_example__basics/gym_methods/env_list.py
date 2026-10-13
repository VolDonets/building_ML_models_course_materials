from gymnasium import envs


for e in envs.registry.values():
    print(e.id)
    