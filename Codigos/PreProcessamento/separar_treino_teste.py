# -*- coding: utf-8 -*-
"Bruna Almeida Osti - 1829009"

import split_folders

split_folders.ratio('eye',\
                    output="eye_dataset",\
                    seed=666,\
                    ratio=(.8, .0, .2)) #separando a pasta em 80% treinamento, 0% validação e 20% treinamento