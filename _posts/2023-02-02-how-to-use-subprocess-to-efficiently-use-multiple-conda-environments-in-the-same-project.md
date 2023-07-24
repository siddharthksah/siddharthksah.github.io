---
title: ' How to use subprocess to efficiently use multiple conda environments in the same project'
date: 2023-02-02
permalink: /posts/2023/02/how-to-use-subprocess-to-efficiently-use-multiple-conda-environments-in-the-same-project/
tags:
  - Conda
  - Multi Environment
  - Subprocess
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
            In this article, we will explore how subprocess can help you work with different conda environments within your project easily. This approach enhances productivity and makes it simpler to manage your code effectively.
        </p>
    </div>
    <div class="intro-image">
        <img src="https://github.com/siddharthksah/siddharthksah.github.io/blob/master/_posts/2023-02-02-how-to-use-subprocess-to-efficiently-use-multiple-conda-environments-in-the-same-project/how-to-use-subprocess-to-efficiently-use-multiple-conda-environments-in-the-same-project.jpeg?raw=true">
        <p class="image-caption"><em>Image generated using text-to-image model by Adobe</em></p>
    </div>
</div> 


Let's start with the most basic method of managing conda environments.

  1. Create separate environments for each project requirement:

    
    
    conda create -n env1 python=x.x   
    conda create -n env2 python=y.y

2\. Activate the desired environment before working on the project:

    
    
    conda activate env1

3\. Deactivate the current environment before switching to another:

    
    
    conda deactivate

4\. Export the environment’s specifications to a file:

    
    
    conda env export > env1.yml

By doing this, you can have different packages and dependencies for each
environment, and switch between them easily as needed through the _terminal_.

This works fine unless you need to do this on the fly, i.e., between different
conda environments for the same project.

Multiple conda environments in a Python project can be necessary for several
reasons, including:

  1. Package management: Different projects may require different versions of packages or different packages altogether. Having separate conda environments ensures that each project has the specific packages it requires and reduces the risk of version conflicts.
  2. Reproducibility: Separate environments ensure that the project can be easily run on different machines and that results can be reproduced.
  3. Isolation: Different environments can be isolated from one another, which is useful for testing and development.
  4. Compatibility: By using separate conda environments, you can create an environment that is compatible with the specific requirements of a project, such as a specific Python version.

In short, using multiple conda environments provides greater control and
organization for Python projects and reduces the risk of conflicts and
compatibility issues.

The below code defines a function `switch_conda_environment` which takes the
name of a conda environment as an argument and switches to that environment
using the `subprocess.run` function. The `shell=True` argument is used to run
the command in a shell, which is required to use the `conda` command. The
function prints the name of the environment that was switched to, which can be
useful for debugging purposes.

In the example usage section, the function is used to switch between two
different environments, “env1” and “env2”. You can replace these names with
the names of your own conda environments.

    
    
    import subprocess  
      
    def switch_conda_environment(env_name):  
        subprocess.run(f"conda activate {env_name}", shell=True)  
        print(f"Switched to conda environment: {env_name}")  
      
    # Example usage:  
    switch_conda_environment("env1")  
    switch_conda_environment("env2")

For deployment, I advise adding another function that checks if the
environment exists else creates one for you with the given requirements. You
can go crazy and handle errors too. Just purge and reinstall. For demo I am
going to stick to the fundamentals.

    
    
    import subprocess  
      
    def create_conda_environment(env_name, requirements_file):  
        env_exists = False  
        try:  
            subprocess.run(f"conda activate {env_name}", shell=True, check=True)  
            env_exists = True  
        except subprocess.CalledProcessError as e:  
            pass  
          
        if not env_exists:  
            subprocess.run(f"conda create --name {env_name} --file {requirements_file}", shell=True)  
            print(f"Conda environment {env_name} created.")  
        else:  
            print(f"Conda environment {env_name} already exists.")  
      
    # Example usage:  
    create_conda_environment("env1", "requirements.txt")

This code defines a function ``create_conda_environment` which takes the name
of a conda environment and a requirements file as arguments. The function
first attempts to activate the environment using the `subprocess.run`
function. If the environment does not exist, a `CalledProcessError` is raised,
and the function proceeds to create the environment using the ``conda create`
command and the requirements file. The `shell=True` argument is used to run
the command in a shell, which is required to use the `conda` command. The
function prints a message indicating whether the environment was created or
already existed.

In the example usage section, the function is used to create an environment
named “env1” using the requirements file "requirements.txt". You can replace
these values with your own environment name and requirements file.

Having multiple conda environments in the same project also means you have the
flexibility to use different Python versions for different projects. This is a
big relief if you want to work with papers that use different Python versions,
but the Python versions are not compatible. It even works with Python 2. Here
is the code, which also takes the Python version as an argument.

    
    
    import subprocess  
      
    def create_conda_environment(env_name, requirements_file, python_version):  
        env_exists = False  
        try:  
            subprocess.run(f"conda activate {env_name}", shell=True, check=True)  
            env_exists = True  
        except subprocess.CalledProcessError as e:  
            pass  
          
        if not env_exists:  
            subprocess.run(f"conda create --name {env_name} python={python_version} --file {requirements_file}", shell=True)  
            print(f"Conda environment {env_name} created.")  
        else:  
            print(f"Conda environment {env_name} already exists.")  
      
    # Example usage:  
    create_conda_environment("env1", "requirements.txt", "3.9")

In this updated code, the function `create_conda_environment` takes a third
argument `python_version`, which specifies the desired version of Python for
the conda environment. This version is passed to the `conda create` command
using the `python` option.

In the example usage section, the function is used to create an environment
named “env1” with Python version 3.9 and the requirements file
“requirements.txt”. You can replace these values with your own environment
name, requirements file, and desired Python version.

Here is a master function which puts all of this together and works seamlessly
when working with multiple environments.

    
    
    import subprocess  
      
    def handle_conda_environment(env_name, requirements_file, python_version):  
        def create_conda_environment():  
            subprocess.run(f"conda create --name {env_name} python={python_version} --file {requirements_file}", shell=True)  
            print(f"Conda environment {env_name} created.")  
      
        def switch_conda_environment():  
            subprocess.run(f"conda activate {env_name}", shell=True)  
            print(f"Switched to conda environment: {env_name}")  
      
        env_exists = False  
        try:  
            switch_conda_environment()  
            env_exists = True  
        except subprocess.CalledProcessError as e:  
            pass  
          
        if not env_exists:  
            create_conda_environment()  
        else:  
            print(f"Conda environment {env_name} already exists.")  
      
    # Example usage:  
    handle_conda_environment("env1", "requirements.txt", "3.9")

This code defines a function `handle_conda_environment` which takes the name
of a conda environment, a requirements file, and a Python version as
arguments. The function contains two inner functions
`create_conda_environment` and `switch_conda_environment` which perform the
respective tasks as described in previous answers.

The `handle_conda_environment` function first attempts to switch to the
specified environment using the `switch_conda_environment` function. If the
environment does not exist, a `CalledProcessError` is raised, and the function
proceeds to create the environment using the `create_conda_environment`
function. The function prints messages indicating the state of the
environment, whether it was created, switched to, or already existed.

In the example usage section, the function is used to handle an environment
named “env1” with Python version 3.9 and the requirements file
“requirements.txt”. You can replace these values with your own environment
name, requirements file, and desired Python version.