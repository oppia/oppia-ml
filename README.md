# Oppia-ml

Oppia-ml is a supplementary component which is used with Oppia for training machine learning models on a separate VM instance. Oppia puts job requests for training a classifier in database. Oppia-ml picks this job requests one by one, trains classifier for these requests and stores the result of training back into database. Oppia uses this stored results to predict outcome for answers.
 
Oppia-ml is written in Python and uses various machine learning libraries for creating classifiers.

## Installation

### Installing on your development machine

1. Clone this repo in folder oppia-ml.

2. Open terminal and navigate to oppia-ml folder and run:
  ```
    git checkout develop
    bash scripts/start.sh
  ```

 
### Deploying Oppia-ml on VM instance

1. Clone this repo in oppia-ml folder of your VM instance.

2. Install Supervisor on VM instance. Generally it can be installed by running simple pip command. You need superuser privileges to install it on VM. if this command does not work then follow instructions on official installation page. 
  ```
    pip install supervisor
  ```

3. Navigate to oppia-ml folder in terminal and run following commands:
  ```
    bash scripts/deploy.sh
  ```

4. Add shared secret key in VM and in Oppia for secure communication.
  Shared key on VM is added using GCE metadata. Add two key - value pairs in metadata, one for “shared_secret_key” and other is “vm_id”. VM will automatically get the ID and secret from metadata.
  Shared key on Oppia can be added by going to “/admin” page of your Oppia host. On this page go to “config” tab where there will be one section for VMID and shared secret keys in which one can add as many “vm_id” and “shared_secret_key” as needed.

## Support
If you have any feature requests or bug reports, please log them on our [issue tracker](https://github.com/oppia/oppia-ml/issues/new?title=Describe%20your%20feature%20request%20or%20bug%20report%20succinctly&body=If%20you%27d%20like%20to%20propose%20a%20feature,%20describe%20what%20you%27d%20like%20to%20see.%20Mock%20ups%20would%20be%20great!%0A%0AIf%20you%27re%20reporting%20a%20bug,%20please%20be%20sure%20to%20include%20the%20expected%20behaviour,%20the%20observed%20behaviour,%20and%20steps%20to%20reproduce%20the%20problem.%20Console%20copy-pastes%20and%20any%20background%20on%20the%20environment%20would%20also%20be%20helpful.%0A%0AThanks!).
 
Please report security issues directly to admin@oppia.org.
 
## Licence
The Oppia-ml code is released under the [Apache v2 license](https://github.com/oppia/oppia-ml/blob/master/LICENSE).
