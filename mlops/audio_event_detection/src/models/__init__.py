# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022 STMicroelectronics.
#  * All rights reserved.
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

from .custom_model import get_model
from .model_utils import add_head 
from .yamnet import yamnet
from .miniresnet import miniresnet
from .miniresnetv2 import miniresnetv2