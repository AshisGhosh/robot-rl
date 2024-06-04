default:
    just -l

train env_id="PickCube-v1":
    xhost +
    docker compose run train-dry-run poetry run python robot_rl/train_robot.py env.id={{env_id}}
    xhost -

train-dry-run env_id="PickCube-v1":
    xhost +
    docker compose run train-dry-run poetry run python robot_rl/train_robot.py dry_run=True env.id={{env_id}}
    xhost -

visualize-train:
    xhost +
    docker compose up visualize-train --build
    xhost -

visualize-env env_id="PickCube-v1":
    xhost +
    docker compose run visualize-env poetry run python robot_rl/visualize_env.py --env_id {{env_id}}
    xhost -

eval:
    xhost +
    docker compose up eval --build
    xhost -

maniskill-demo:
    xhost +
    docker compose up maniskill-demo --build
    xhost -

make-owner:
    sudo chown -R 1000:1000 .

python-test:
    #!/usr/bin/env python3
    print("hello world")