default:
    just -l

train:
    xhost +
    docker compose up train --build
    xhost -

visualize-train:
    xhost +
    docker compose up visualize-train --build
    xhost -

visualize-env:
    xhost +
    docker compose up visualize-env --build
    xhost -

python-test:
    #!/usr/bin/env python3
    print("hello world")