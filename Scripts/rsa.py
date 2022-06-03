import numpy as np
import rsatoolbox as rsa

def get_rdms(bcioutVCI, bcioutACI, bcioutposteriorCI, # for CI model
             bcioutVFF, bcioutAFF, bcioutposteriorFF, # for FF model
             bcioutVFS, bcioutAFS, bcioutposteriorFS, # for FS model
             nnoutV, nnoutA, # for NN behav data
             nnoutactivations, # for NN activation data -- this will change as architectures change, so future proof
             conditions): # conditions to get groundtruth rdm

    pass