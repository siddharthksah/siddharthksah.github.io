---
title: 'Automating SSH Login and Jupyter Notebook Setup for Machine Learning Projects'
date: 2023-03-07
permalink: /posts/2023/03/automating-ssh-login-and-jupyter-notebook-setup-for-machine-learning-projects/
tags:
  - SSH
  - Automation
  - Jupyter Notebook
---
<style>
    .blog-intro {
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-family: Arial, sans-serif;
        font-size: 16px;
        line-height: 1.6;
        color: #333;
        background-color: #f8f8f8;
        padding: 20px;
    }

    .intro-text {
        flex: 1;
        font-weight: bold;
    }

    .intro-image {
        flex: 1;
        margin-left: 20px;
        text-align: right;
    }

    .intro-image img {
        width: 100%;
        border-radius: 8px;
    }

    .image-caption {
        font-size: 8px;
        color: #666;
        margin-top: 0px;
    }
</style>

<div class="blog-intro">
    <div class="intro-text">
        <p>
            In this article, we delve into streamlining the setup process for machine learning projects by automating SSH login and Jupyter Notebook configuration. By automating these crucial tasks, developers can save valuable time and effortlessly access their projects, fostering a more efficient and productive workflow.
        </p>
    </div>
    <div class="intro-image">
        <img src="https://github.com/siddharthksah/siddharthksah.github.io/blob/master/posts/automating-ssh-login-and-jupyter-notebook-setup-for-machine-learning-projects.jpeg?raw=true">
        <p class="image-caption"><em>Image generated using text-to-image model by Adobe</em></p>
    </div>
</div>

SSH Login: SSH (Secure Shell) is a cryptographic network protocol for operating network services securely over an unsecured network. It is commonly used for remote command-line login and remote command execution. To automate SSH login, we will use the Paramiko library.

Paramiko is a Python implementation of the SSH protocol. It allows you to create SSH connections and execute commands on the remote server. In our script, we get the SSH ID, password, Python version, Conda flag, and venv flag as command-line arguments. We also check if Paramiko is installed and install it if it is not installed.

Jupyter Notebook is an open-source web application that allows you to create and share documents that contain live code, equations, visualizations, and narrative text. It is widely used in the machine learning community. To automate Jupyter Notebook setup, we will use the following steps:

1.  Check if Conda is installed on the remote server, and install it if it does not exist.
2.  Create a new Conda environment with the specified Python version if the Conda flag is set to True.
3.  Create a new Python virtual environment using venv if the venv flag is set to True.
4.  Install Jupyter Notebook and create a new configuration.
5.  Set the password for the Jupyter Notebook server.
6.  Open a tunnel to the remote server using SSH port forwarding.
7.  Start the Jupyter Notebook server and open it in a web browser.
8.  Wait for the user to exit the Jupyter Notebook.
9.  Kill the SSH tunnel process and close the SSH connection.

Automating SSH login and Jupyter Notebook setup for machine learning projects is a great way to save time and increase productivity. It allows you to focus on developing your machine learning models instead of spending time on server setup and configuration. In this article, we have discussed how to automate SSH login and Jupyter Notebook setup using Python. By following these steps, you can easily set up your machine learning project on a remote server and start developing your models.

```python
import sys
import os
import subprocess
import webbrowser

# Check if paramiko is installed
try:
    import paramiko
except ImportError:
    print("Error: Paramiko is not installed.")
    print("Please install paramiko using the following command:")
    print("pip install paramiko")
    sys.exit(1)

# Rest of the code remains the same...


# Get the SSH ID, password, Python version, Conda flag, and venv flag as command line arguments  
if len(sys.argv) != 6:  
    print("Usage: python ssh_login.py <ssh_id> <password> <python_version> <conda> <venv>")  
    sys.exit(1)  
ssh_id = sys.argv[1]  
password = sys.argv[2]  
python_version = sys.argv[3]  
conda_flag = sys.argv[4].lower() == 'true'  
venv_flag = sys.argv[5].lower() == 'true'  

# Check if paramiko is installed and install it if it is not installed  
try:  
    import paramiko  
except ImportError:  
    print("Paramiko is not installed. Installing Paramiko...")  
    os.system("pip install paramiko")  

# Create an SSH client object  
ssh = paramiko.SSHClient()  

# Set the policy for the client object to auto add the hostname and key to known_hosts file  
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())  

# Connect to the SSH server using the provided hostname, port, username, and password  
try:  
    ssh.connect(hostname='<hostname>', port=<port>, username=ssh_id, password=password)  
    print("Successfully logged in to the SSH server!")  
except paramiko.AuthenticationException:  
    print("Authentication failed. Please check your credentials.")  
    sys.exit(1)  

# Check if Conda is installed on the remote server, and install it if it does not exist  
if conda_flag:  
    conda_check_cmd = "conda --version"  
    stdin, stdout, stderr = ssh.exec_command(conda_check_cmd)  
    output = stdout.read().decode()  
    if "conda: command not found" in output:  
        print("Conda is not installed on the remote server. Installing Conda...")  
        conda_install_cmd = "wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && bash ~/miniconda.sh -b -p $HOME/miniconda"  
        stdin, stdout, stderr = ssh.exec_command(conda_install_cmd)  
        print(stdout.read().decode())  
    else:  
        print("Conda is already installed on the remote server.")  

    # Install nb_conda  
    nb_conda_install_cmd = "conda install nb_conda -y"  
    stdin, stdout, stderr = ssh.exec_command(nb_conda_install_cmd)  
    print(stdout.read().decode())  

# Create a new Conda environment with the specified Python version, if the Conda flag is set to True  
if conda_flag:  
    conda_cmd = f"conda create -n myenv python={python_version}"  
    stdin, stdout, stderr = ssh.exec_command(conda_cmd)  
    print(stdout.read().decode())  

    # Activate the new Conda environment  
    activate_cmd = "conda activate myenv"  
    stdin, stdout, stderr = ssh.exec_command(activate_cmd)  

# Create a new Python virtual environment using venv, if the venv flag is set to True  
if venv_flag:  
    venv_cmd = f"python{python_version} -m venv myenv"  
    stdin, stdout, stderr = ssh.exec_command(venv_cmd)  
    print(stdout.read().decode())  

    # Activate the new virtual environment  
    activate_cmd = "source myenv/bin/activate"  
    stdin, stdout, stderr = ssh.exec_command(activate_cmd)  

# Install Jupyter Notebook and create a new configuration  
install_jupyter_cmd = "pip install jupyter"  

stdin, stdout, stderr = ssh.exec_command(install_jupyter_cmd)  
print(stdout.read().decode())  

# Generate a new configuration file for Jupyter Notebook  
generate_config_cmd = "jupyter notebook --generate-config"  
stdin, stdout, stderr = ssh.exec_command(generate_config_cmd)  
print(stdout.read().decode())  

# Set the password for the Jupyter Notebook server  
password_hash = subprocess.check_output(['python', '-c', "from notebook.auth import passwd; print(passwd('" + password + "'))"]).decode().strip()  

config_file = os.path.expanduser("~/.jupyter/jupyter_notebook_config.py")  
with open(config_file, 'a') as f:  
    f.write("\n")  
    f.write("# Set the password for the Jupyter Notebook server\n")  
    f.write(f"c.NotebookApp.password = '{password_hash}'\n")  

# Open a tunnel to the remote server using SSH port forwarding  
localhost = 'localhost'  
port = 8888  
remote_port = 8888  

tunnel_cmd = f"ssh -N -L {port}:{localhost}:{remote_port} {ssh_id}@<hostname>"  

# Start the SSH tunnel process  
tunnel_proc = subprocess.Popen(tunnel_cmd, shell=True)  

# Start the Jupyter Notebook server and open it in a web browser  
start_jupyter_cmd = f"jupyter notebook --no-browser --port={remote_port}"  
stdin, stdout, stderr = ssh.exec_command(start_jupyter_cmd)  
print(stdout.read().decode())  

webbrowser.open_new_tab(f'http://{localhost}:{port}/')  

# Wait for the user to exit the Jupyter Notebook  
print("Press ENTER to close the SSH tunnel and exit the program.")  
input()  

# Kill the SSH tunnel process  
tunnel_proc.kill()  

# Close the SSH connection  
ssh.close()
```

It can be challenging to train and test machine learning models on a single computer due to the limited computational resources. Therefore, it is often necessary to use remote servers or cloud computing platforms to run machine learning experiments. In this article, we explored a Python script that facilitates logging into a remote server via SSH, setting up a machine learning environment, and running Jupyter Notebook.

The code above starts with importing several libraries, including paramiko, os, subprocess, and webbrowser. The paramiko library is a Python implementation of the SSH protocol and can be used to create an SSH client object. The os library provides a way to interact with the operating system, and the subprocess library can be used to execute shell commands from Python. Finally, the webbrowser library provides an interface to interact with web browsers.

The next section of the code gets the command-line arguments for the SSH ID, password, Python version, Conda flag, and venv flag. The Conda flag is used to specify whether Conda should be installed, and the venv flag is used to specify whether a Python virtual environment should be created. The argparse library can also be used to parse command-line arguments in Python.

After getting the command-line arguments, the code checks if the paramiko library is installed and installs it if it is not installed. This is done using the os.system() method to execute the pip install command. Once the paramiko library is installed, an SSH client object is created using the paramiko.SSHClient() method.

The next step is to set the policy for the client object to auto add the hostname and key to known\_hosts file using the ssh.set\_missing\_host\_key\_policy(paramiko.AutoAddPolicy()) method. This ensures that the client object trusts the remote server and does not prompt the user to manually verify the server’s fingerprint.

After setting the policy, the code connects to the remote server using the ssh.connect() method. The hostname, port, SSH ID, and password are provided as arguments. The code also catches the paramiko.AuthenticationException exception to handle authentication errors gracefully.

If the Conda flag is set to True, the code checks if Conda is installed on the remote server using the conda — version command. If Conda is not installed, the code downloads and installs the latest version of Miniconda using the wget command and the bash command to execute the installation script. The nb\_conda package is also installed using the conda install nb\_conda -y command.

If the Conda flag is set to True, the code creates a new Conda environment using the specified Python version using the conda create -n myenv python={python\_version} command. The environment is activated using the conda activate myenv command. If the venv flag is set to True, the code creates a new Python virtual environment using the python -m venv myenv command and activates it using the source myenv/bin/activate command.

The code then installs Jupyter Notebook using the pip install jupyter command and generates a new configuration file using the jupyter notebook — generate-config command. The password for the Jupyter Notebook server is set using the notebook.auth.passwd() method, which generates a SHA-1 hash of the password. The hash is added to the jupyter\_notebook\_config.py file using the f.write() method.

The code then opens an SSH tunnel to the remote server using port forwarding using the ssh -N -L {port}:{localhost}:{remote\_port} command. The tunnel is started using the subprocess.Popen() method, which executes the command in a new process.

Once the tunnel is open, the code starts the Jupyter Notebook server on the remote server with the `start_jupyter_cmd` command. The `--no-browser` option is used to start the server without opening a browser window. Instead, the code opens a new tab in the default web browser of the user with the `webbrowser` module. The URL of the Jupyter Notebook server is `[http://localhost:8888/](http://localhost:8888/.)`[.](http://localhost:8888/.)

After starting the Jupyter Notebook server, the code prompts the user to press ENTER to close the SSH tunnel and exit the program. Once the user presses ENTER, the code kills the SSH tunnel process and closes the SSH connection with the `tunnel_proc.kill()` and `ssh.close()` commands, respectively.

This code is a powerful tool for machine learning professionals who need to work with remote servers to train and deploy machine learning models. By automating the setup of a remote machine, installing the necessary software, and starting a Jupyter Notebook server, this code saves valuable time and resources that can be better spent on developing and improving machine learning models.

Limitations
-----------

The code performs various tasks to automate the setup of a Jupyter Notebook server on a remote server using SSH. However, there are several limitations and potential areas for improvement in the code:

1.  Lack of input validation: The code assumes that the command-line arguments and inputs provided by the user are valid, which can lead to unexpected errors or security issues. For example, the code does not check if the provided hostname or port are valid or if the provided password is strong enough.
2.  Limited error handling: The code does not have robust error handling mechanisms to handle unexpected errors or exceptions that may occur during the execution of the script. For instance, the code assumes that the SSH connection will always be successful, but it does not handle cases where the connection fails due to network issues or incorrect credentials.
3.  Security concerns: The code uses a plaintext password to authenticate with the remote server, which is not secure. Additionally, the code generates and stores the Jupyter Notebook password in plaintext in the configuration file, which is also a security concern.
4.  Compatibility issues: The code assumes that the remote server is running a Linux-based operating system and that the user has administrative privileges to install packages and create environments.

To improve the code, here are some suggestions which can be incorporated in the next version:

1.  Add input validation: The code should validate user inputs and command-line arguments to prevent unexpected errors or security issues. For example, the code can use regular expressions to validate the hostname, port, and password inputs.
2.  Improve error handling: The code should have robust error handling mechanisms to handle unexpected errors or exceptions that may occur during the execution of the script. For instance, the code can use try-except blocks to catch and handle exceptions that may occur during SSH connection or package installation.
3.  Enhance security: The code can use more secure methods for password authentication, such as SSH keys or two-factor authentication. Additionally, the code can use a secure method to store the Jupyter Notebook password, such as the keyring module.
4.  Improve compatibility: The code can check the remote server’s operating system and the user’s privileges before attempting to install packages or create environments. Additionally, the code can use platform-independent package managers such as conda or pipenv instead of system-specific package managers.
5.  Add logging: The code can add logging statements to track the progress and errors during the script’s execution. This can help in debugging and troubleshooting issues that may occur during the script’s execution.
6.  Use command-line arguments for all inputs: Currently, the code uses command-line arguments for the SSH ID, password, Python version, Conda flag, and venv flag. However, the hostname and port are hard-coded in the code. It would be better to use command-line arguments for all inputs so that the user can provide all the necessary information when running the script.
7.  Use context managers for SSH connection: The code currently uses the `connect` method of the `SSHClient` class to establish an SSH connection, but it does not use a context manager to ensure that the connection is properly closed. It would be better to use a `with` statement to create a context manager that automatically closes the connection when the code inside the context is finished.
8.  Use `subprocess` instead of `os.system`: The `os.system` function is deprecated and should be replaced with the `subprocess` module, which provides a more flexible and secure way to run external commands.
9.  Use f-strings for string interpolation: The code currently uses string concatenation and string formatting with the `%` operator. It would be better to use f-strings, which are more readable and less error-prone.
10.  Use more descriptive variable names: Some of the variable names in the code are not very descriptive, which makes it harder to understand what the code is doing. For example, the variable `stdin` is used to store the standard input stream of an SSH command, which is not very clear. It would be better to use more descriptive names like `stdin_stream` or `input_stream`.

I have tried below to incorporate some of them. This is still experiment please use it with caution.

```python
import argparse
import subprocess
import paramiko
import webbrowser
import os
import io

def check_and_install_package(package):
    """
    Check if a package is installed and install it if it is not installed
    """
    try:
        __import__(package)
    except ImportError:
        print(f"{package} is not installed. Installing {package}...")
        subprocess.run(["pip", "install", package])

def ssh_login(hostname, port, ssh_key_content, python_version, use_conda=False, use_venv=False):
    """
    Connects to a remote server using SSH and executes Python code.
    :param hostname: Hostname of the remote server.
    :param port: Port number to connect to the remote server.
    :param ssh_key_content: Content of the SSH private key.
    :param python_version: Version of Python to use on the remote server.
    :param use_conda: Flag indicating whether to use conda as the Python environment manager.
    :param use_venv: Flag indicating whether to use virtual environments as the Python environment manager.
    """
    check_and_install_package("paramiko")

    # Create an SSH client object
    ssh = paramiko.SSHClient()

    # Set the policy for the client object to auto add the hostname and key to known_hosts file
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # Load the private key content
    ssh_key = paramiko.RSAKey(file_obj=io.StringIO(ssh_key_content))

    try:
        # Connect to the SSH server using the provided hostname, port, and SSH key
        ssh.connect(hostname=hostname, port=port, username=os.getlogin(), pkey=ssh_key)

        # Execute the Python code on the remote server
        command = f"python{python_version}"
        if use_conda:
            command = f"conda run -n my_env {command}"
        elif use_venv:
            command = f"source my_env/bin/activate && {command}"
        stdin, stdout, stderr = ssh.exec_command(command)

        # Print the output of the command
        for line in stdout:
            print(line.strip())

        # Print the errors, if any
        for line in stderr:
            print(line.strip())

    except paramiko.ssh_exception.AuthenticationException:
        print("Authentication failed. Please check your username and password.")
    except paramiko.ssh_exception.NoValidConnectionsError:
        print("Unable to connect to the server. Please check the hostname.")
    except Exception as e:
        print(f"An error occurred while connecting to the server: {e}")
    finally:
        # Close the SSH connection
        ssh.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SSH Login and Jupyter Notebook Setup")
    parser.add_argument("hostname", type=str, help="Hostname of the remote server")
    parser.add_argument("port", type=int, help="Port number to connect to the remote server")
    parser.add_argument("ssh_key", type=str, help="Path to the SSH private key file")
    parser.add_argument("python_version", type=str, help="Version of Python to use on the remote server")
    parser.add_argument("--use_conda", action="store_true", help="Flag indicating whether to use conda")
    parser.add_argument("--use_venv", action="store_true", help="Flag indicating whether to use virtual environments")
    args = parser.parse_args()

    with open(args.ssh_key, "r") as f:
        ssh_key_content = f.read()

    ssh_login(args.hostname, args.port, ssh_key_content, args.python_version, args.use_conda, args.use_venv)
```
