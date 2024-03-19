# Turbine Tank

## Setup

```
git clone https://github.com/nod-ai/SHARK-Turbine
```

Now, go back to the TestSuite Repo, and create a python environment in TestSuite/e2eshark

```
pip install -f https://openxla.github.io/iree/pip-release-links.html --upgrade -r 'your local SHARK Turbine repo'/core/iree-requirements.txt
pip install -e 'your local SHARK Turbine repo'/core[testing]
pip install -e 'your local SHARK Turbine repo'/models
```

## Overview

Turbine tank allows us to currently run llama, sd models, and 30 other models e2e and upload torch mlir artifacts to Azure. Both inline weights and external parameter versions are uploaded (take a look at `classic_flow` and `param_flow` in tank_util.py).

If interested in how the azure side of uploading and downloading while maintaining versioning (date + git_sha) please take a look at this file: [azure handling](https://github.com/nod-ai/SHARK-Turbine/blob/main/models/turbine_models/turbine_tank/turbine_tank.py), which is a part of the turbine changes needed for turbine tank to work.

There currently is a folder uploaded into Azure tankturbine storage container using turbine tank if you are interested in structure and how we keep track of the version using the date + git_sha: [container_link](https://portal.azure.com/#view/Microsoft_Azure_Storage/ContainerMenuBlade/~/overview/storageAccountId/%2Fsubscriptions%2F8c190d1b-eb91-48d5-bec5-3e7cb7412e6c%2FresourceGroups%2Fpdue-nod-ai-rg%2Fproviders%2FMicrosoft.Storage%2FstorageAccounts%2Ftankturbine/path/tankturbine/etag/%220x8DC2CE680A9B29E%22/defaultEncryptionScope/%24account-encryption-key/denyEncryptionScopeOverride~/false/defaultId//publicAccessVal/Container). The torch IR for every model is uploaded when the user choses to upload using turbine tank. Accuracy against a torch run is also tested for every model we can run e2e. This allows us to keep track of the accuracy of our models and avoid regressions. 

There is also an option to download turbine tank model artifacts locally, so you don't have to run the models. It is setup so that it will not redownload and use cached artifacts if already there. It will also update your local artifacts if they are not up to date. 

Further detail can be found in the code comments and the turbine changes in SHARK-Turbine (llama tests, sd tests, custom models, azure handling).

To run turbine tank:
Build turbine like normal in a python venv.
Run `python turbine_tank/run_tank.py`
Run `python turbine_tank/run_tank.py --download_ir` to download mlir from azure to local cache

But, a user won't be able to run this. There are environment variables that need to be configured to connect to Azure. We only want our nightly job uploading into azure. 
