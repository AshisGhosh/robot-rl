default:
    just -l

train:
    xhost +
    docker compose up train --build
    xhost -

train-dry-run:
    xhost +
    docker compose up train-dry-run --build
    xhost -

visualize-train:
    xhost +
    docker compose up visualize-train --build
    xhost -

visualize-env:
    xhost +
    docker compose up visualize-env --build
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