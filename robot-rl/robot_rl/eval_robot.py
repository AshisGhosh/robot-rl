import os
import sys
import subprocess
import hydra
from omegaconf import DictConfig
import select
from eval_sb3 import eval_sb3
import logging


@hydra.main(version_base="1.3", config_path="../conf", config_name="eval_config")
def main(cfg: DictConfig):
    policy_library = cfg.policy.library
    env_library = cfg.env.library

    if policy_library == "maniskill" or env_library == "maniskill":
        output_dir = hydra.core.hydra_config.HydraConfig.get().run.dir

        assert (
            policy_library == env_library
        ), "Policy and environment libraries must match if using maniskill"
        # Run the training script with unbuffered output
        command = [
            sys.executable,
            "-u",  # Add -u for unbuffered output
            "robot_rl/maniskill_ppo.py",
            f"--output_dir={output_dir}",
            "--evaluate",
            f"--env_id={cfg.env.id}",
            f"--checkpoint={cfg.model_path}",
            "--num_eval_envs=1",
            f"--num-eval-steps={cfg.env.num_eval_steps}",
            f"--sim_backend={cfg.env.sim_backend}",
            f"--render_mode={cfg.env.eval_render_mode}",
            f"--shader={cfg.env.eval_shader}",
        ]

        try:
            # Run the training script and stream the output in real-time
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )

            # Use select to monitor both stdout and stderr
            while True:
                reads = [process.stdout.fileno(), process.stderr.fileno()]
                ret = select.select(reads, [], [])

                for fd in ret[0]:
                    if fd == process.stdout.fileno():
                        output = process.stdout.readline()
                        if output:
                            print(output.strip())
                    if fd == process.stderr.fileno():
                        error_output = process.stderr.readline()
                        if error_output:
                            print(error_output.strip(), file=sys.stderr)

                if process.poll() is not None:
                    break

            # Ensure all output is read
            while True:
                output = process.stdout.readline()
                if output == "":
                    break
                print(output.strip())

            while True:
                error_output = process.stderr.readline()
                if error_output == "":
                    break
                print(error_output.strip(), file=sys.stderr)

            # Wait for the process to finish and get the return code
            return_code = process.wait()
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, command)

        except subprocess.CalledProcessError as e:
            # Handle errors in the called script
            print(f"Error: {e}", file=sys.stderr)

    if policy_library == "sb3":
        # Set up logging
        log_file = os.path.join(
            hydra.core.hydra_config.HydraConfig.get().run.dir, "evaluation_log.txt"
        )
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        logger = logging.getLogger()
        file_handler = logging.FileHandler(log_file)
        logger.addHandler(file_handler)
        eval_sb3(cfg=cfg, logger=logger)


if __name__ == "__main__":
    main()
