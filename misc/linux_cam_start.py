import subprocess


def linux():

    print("running module loader")

    #remove previous kernel Module
    try:
        subprocess.run(['sudo', 'modprobe', '-r', 'v4l2loopback'], check=True)
        print("Previous Module Removed Successfully")
    except subprocess.CalledProcessError:
        print("No previous Module detected")

    # Load v4l2loopback kernel module
    try:
        subprocess.run([
            'sudo', 'modprobe', 'v4l2loopback',
            'devices=1',
            'video_nr=10',
            'card_label="WebCam"',
            'exclusive_caps=1'
        ], check=True)
        print("v4l2loopback module loaded successfully.")
    except subprocess.CalledProcessError:
        print("Failed to load v4l2loopback module. It may already be loaded or require sudo access.")


if __name__ == "__main__":
    linux()