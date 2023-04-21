import numpy as np
from abc import abstractmethod
from utils.cython.im2col import col2im, im2col
from utils.activations import get_activation


class Conv2D:
    def __init__(self, num_filters,
                 filter_size=3,
                 channels=1,
                 stride=1,
                 padding=0,
                 weight_scale=1e-3,
                 activation='identity'):
        """
        Keyword Arguments:
            num_filters {int} -- nombre de cartes d'activation.
            filter_size {int, tuple} -- taille des filtres. (default: {3})
            channels {int} -- nombre de canaux. Doit être égal au nombre
                              de canaux des données en entrée. (default: {1})
            stride {int, tuple} -- taille de la translation des filtres. (default: {1})
            padding {int, tuple} -- nombre de zéros à rajouter avant et
                                    après les données. La valeur représente
                                    seulement les zéros d'un côté. (default: {0})
            weight_scale {float} -- écart type de la distribution normale utilisée
                                    pour l'initialisation des weights. (default: {1e-4})
            activation {str} -- identifiant de la fonction d'activation de la couche
                                (default: {'identite'})
        """

        self.num_filters = num_filters
        self.filter_size = filter_size
        self.channels = channels
        self.weight_scale = weight_scale
        self.activation_id = activation
        
        if isinstance(stride, tuple):
            self.stride = stride
        elif isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            raise Exception("Invalid stride format, must be tuple or integer")

        if isinstance(padding, tuple):
            self.pad = padding
        elif isinstance(padding, int):
            self.pad = (padding, padding)
        else:
            raise Exception("Invalid padding format, must be tuple or integer")

        if not isinstance(channels, int):
            raise Exception("Invalid channels format, must be integer")

        if isinstance(filter_size, tuple):
            self.W = np.random.normal(loc=0.0, scale=weight_scale, size=(num_filters, channels, filter_size[0],
                                                                         filter_size[1]))
        elif isinstance(filter_size, int):
            self.W = np.random.normal(loc=0.0, scale=weight_scale, size=(num_filters, channels, filter_size,
                                                                         filter_size))
        else:
            raise Exception("Invalid filter format, must be tuple or integer")

        self.b = np.zeros(num_filters)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)
        self.reg = 0.0
        self.cache = None

        self.activation = get_activation(activation)

    @abstractmethod
    def forward(self, X, **kwargs):
        pass

    @abstractmethod
    def backward(self, dA, **kwargs):
        pass

    def get_params(self):
        return {'W': self.W, 'b': self.b}

    def get_gradients(self):
        return {'W': self.dW, 'b': self.db}


class Conv2DNaive(Conv2D):

    def forward(self, X, **kwargs):
        """Effectue la propagation avant naïvement.

        Arguments:
            X {ndarray} -- Input de la couche. Shape (N, C, H, W)

        Returns:
            ndarray -- Scores de la couche
        """

        N, channel, height, width = X.shape
        F, Fchannel, Fheight, Fwidth = self.W.shape
        
    	#############################################################################
    	# TODO: Implémentez la propagation pour la couche de convolution.           #
    	# Astuces: vous pouvez utiliser la fonction np.pad pour le remplissage.     #	
    	# NE COPIEZ PAS LES FONCTIONS CYTHONISÉES ET MATRICISÉES                    #
    	#############################################################################

        # remplacer la ligne suivante par du code de convolution
        # Vous devriez avoir besoin de plusieurs boucles imbriquées pour implémenter cette méthode
        # Et n'oubliez pas d'appliquer la fonction d'activation à la fin!

        #On calcule la taille de notre carte d'activation avec les formules du cours
        #Pourquoi on mets +2*(padding) ?
        #Car on a une taille de matrice X + 2*padding
        Ax = int((height+(2*self.pad[0]) - Fheight)/self.stride[0]) +1
        Ay = int((width+(2*self.pad[1]) - Fwidth)/self.stride[1]) +1
        A = np.zeros([N,F,Ax,Ay])

        #print(A.shape)
        #print(height-Fheight)
        #print(width-Fwidth)

        #On boucle sur les échantillons
        for echantillon in range(N):
            #On boucle sur les cartes d'activation
            for carteActivation in range(F):
                #On va traiter les canaux
                for channelToCompute in range(channel):

                    #Convolution initiale sur un signal XxY avec un filtre FX x FY

                    #On boucle sur les lignes
                    for line in range(Ax):
                        #On boucle sur les colonnes
                        for column in range(Ay):
                            #On initialise la valeur à ajouter à 0
                            filterValue = 0

                            #On parcours le filtre sur toutes ses valeurs
                            #Filtre : ligne
                            for filterline in range(Fheight):
                                #Filtre : colonne
                                for filtercolumn in range(Fwidth):
                                    #print("X Line : ",line+filterline)
                                    #print("Y Column : ",column+filtercolumn)

                                    #On calcule la valeur au niveau de la ligne de l'échantillon
                                    #Pourquoi cette formule ?

                                    ###On parcours notre matrice finale et pour chaque valeur, on a
                                    #la matrice finale * stride pour savoir où commencer à prendre
                                    #les valeurs dans notre matrice d'échantillons

                                    ###On ajoute filterline pour faire le produit en notre filtre et
                                    #notre bonne valeur d'échantilllon

                                    ###On retranche le padding pour se positionner au bon endroit dans
                                    #notre matrice X avec le padding
                                    Xline = (line*self.stride[0])+filterline-self.pad[0]

                                    #On calcule la valeur au niveau de la colonne de l'échantillon
                                    #Même raisonnement que pour Xline
                                    Ycolumn = (column*self.stride[1])+filtercolumn-self.pad[1]


                                    #On gère ici le padding. Soit un zero-padding, alors tout ce qui
                                    #est en dehors de X prend la valeur 0. On ajoute donc les conditions
                                    #qui s'assurent que notre Xline et Ycolumn sont dans la matrice échantillon
                                    if ((Xline >= 0) and (Xline < height)) and ((Ycolumn >= 0) and (Ycolumn< width)):
                                        #Ajout de la multiplication du filtre avec l'échantillon
                                        filterValue += self.W[carteActivation][channelToCompute][filterline,filtercolumn] * X[echantillon][channelToCompute][Xline,Ycolumn]
                            
                            #On fait un += pour sommer sur toutes les cartes d'activation
                            #Problème : le biais est ajouté channel fois
                            #Solution : on va l'ajouter qu'au channel == 0 comme ca, on le retire du reste
                            if channelToCompute == 0:
                                A[echantillon,carteActivation,line,column] = filterValue + self.b[carteActivation]
                            else:
                                A[echantillon,carteActivation,line,column] += filterValue
 
        #On n'oublie pas l'activation :)
        A = self.activation['forward'](A)

        #On ajoute le padding car il le faut pour la backpropagation
        X_pad = np.zeros([N, channel, height+(2*self.pad[0]), width+(2*self.pad[1])])
        X_pad[:,:,self.pad[0]:(self.pad[0]+height),self.pad[1]:(self.pad[1]+width)] = X

        self.cache = [X_pad , A, height, width]
        return A

    def backward(self, dA, **kwargs):
        """Effectue la rétropropagation

        Arguments:
            dA {ndarray} -- Dérivée de la loss par rapport au output de la couche.
                            Shape (N, F, out_height, out_width)

        Returns:
            ndarray -- Dérivée de la loss par rapport au input de la couche.
        """
        
        X_col, out, height, width = self.cache

        N, F, out_height, out_width = dA.shape
        _, Fchannel, Fheight, Fwidth = self.W.shape
        dX = np.zeros((N, Fchannel, height, width))

	#############################################################################
	# TODO: Implémentez la rétropropagation pour la couche de convolution       #
	# NE COPIEZ PAS LES FONCTIONS CYTHONISÉES ET MATRICISÉES                    #
	#############################################################################
	
        # Vous devriez avoir besoin de plusieurs boucles imbriquées pour implémenter cette méthode  
        # print(dA)
        # print(self.db.shape)
        # print(self.cache)

        Ax = int((height+(2*self.pad[0]) - Fheight)/self.stride[0]) +1
        Ay = int((width+(2*self.pad[1]) - Fwidth)/self.stride[1]) +1

        # print("--------------")
        # print("--------------")
        # print("dA = ")
        # print(dA.shape)
        # print(Ax)
        # print(Ay)
        # print("--------------")
        # print("Xcol = ")
        # print(X_col.shape)
        # print(((Ax-1)*self.stride[0]))
        # print(((Ax-1)*self.stride[0])+Fheight-1)
        # print(self.pad)
        # print(((Ax-1)*self.stride[0])+Fheight-1+self.pad[0])
        # print(((Ay-1)*self.stride[1])+Fwidth-1+self.pad[1])
        # print("--------------")
        # print("--------------")


        #Ce code est inspiré de celui se trouvant sur le site https://becominghuman.ai/back-propagation-in-convolutional-neural-networks-intuition-and-code-714ef1c38199
        #On commence par itérer sur les dimensions de la sortie
        
        # print(self.W.shape)
        #On boucle sur les échantillons
        for echantillon in range(N):
            #On boucle sur les cartes d'activation
            for carteActivation in range(F):
                #On va traiter les canaux
                for channelToCompute in range(Fchannel):
                    for line in range(int(Ax)):
                        for column in range(int(Ay)):

                            #Ensuite, on va parcourir le filtre pour réaliser la dérivée de la convolution
                            for filterLine in range(Fheight):
                                for filterColumn in range(Fwidth):
                                    #De la même manière que pour la forward, on créé Xline et Ycolumn
                                    Xline = (line*self.stride[0])+filterLine-self.pad[0]
                                    Ycolumn = (column*self.stride[1])+filterColumn-self.pad[1]

                                    #On code les fonctions qui existe dans le site présenté ci-dessus
                                    if ((Xline >= 0) and (Xline < height)) and ((Ycolumn >= 0) and (Ycolumn< width)):
                                        dX[echantillon][channelToCompute][Xline,Ycolumn] += self.W[carteActivation][channelToCompute][filterLine,filterColumn] * dA[echantillon][carteActivation][line,column]
                            
                            #De la même manière que pour la forward, on créé Xline et Ycolumn
                            XBline = (line*self.stride[0])
                            YBcolumn = (column*self.stride[1])
                            XEline = (line*self.stride[0])+Fheight
                            YEcolumn = (column*self.stride[1])+Fwidth

                            self.dW[carteActivation][channelToCompute][:,:] += X_col[echantillon][channelToCompute][XBline:XEline,YBcolumn:YEcolumn] * dA[echantillon][carteActivation][line,column]
                        
                            #dB se mets à jour comme étant la somme des valeurs de dA
                            if channelToCompute == 0:
                                self.db[carteActivation] += dA[echantillon][carteActivation][line,column]

        self.dW += self.reg * self.W
        self.db += self.reg * self.b  

        return dX


class Conv2DMat(Conv2D):

    def forward(self, X, **kwargs):
        """Effectue la propagation en vectorisant.

        Arguments:
            X {ndarray} -- entrée de la couche. Shape (N, C, H, W)

        Returns:
            ndarray -- Scores de la couche
        """

        N, channel, height, width = X.shape
        F, Fchannel, Fheight, Fwidth = self.W.shape

        assert channel == Fchannel
        assert (height - Fheight + 2 * self.pad[0]) % self.stride[0] == 0
        assert (width - Fwidth + 2 * self.pad[1]) % self.stride[1] == 0

        out_height = np.uint32(1 + (height - Fheight + 2 * self.pad[0]) / self.stride[0])
        out_width = np.uint32(1 + (width - Fwidth + 2 * self.pad[1]) / self.stride[1])
        out = np.zeros((N, F, out_height, out_width))

        X_padded = np.pad(X, ((0, 0), (0, 0), (self.pad[0], self.pad[0]), (self.pad[1], self.pad[1])), 'constant')
        
        W_row = self.W.reshape(F, Fchannel*Fheight*Fwidth)

        X_col = np.zeros((N, Fchannel*Fheight*Fwidth, out_height*out_width))
        for index in range(N):
            col = 0
            for i in range(0, height + 2 * self.pad[0] - Fheight + 1, self.stride[0]):
                for j in range(0, width + 2 * self.pad[1] - Fwidth + 1, self.stride[1]):
                    X_col[index, :, col] = X_padded[index, :, i:i+Fheight, j:j+Fwidth]\
                        .reshape(Fchannel*Fheight*Fwidth)
                    col += 1
            out[index] = (W_row.dot(X_col[index]) + self.b.reshape(F, 1)).reshape(F, out_height, out_width)

        self.cache = (X_col, out, height, width)        

        A = self.activation['forward'](out)
        
        return A

    def backward(self, dA, **kwargs):
        """Effectue la rétropropagation en vectorisant.

        Arguments:
            dA {ndarray} -- Dérivée de la loss par rapport à la sortie de la couche.
                            Shape (N, F, out_height, out_width)

        Returns:
            ndarray -- Dérivée de la loss par rapport au input de la couche.
        """

        X_col, out, height, width = self.cache

        N, F, out_height, out_width = dA.shape
        _, Fchannel, Fheight, Fwidth = self.W.shape
        
        pad_height = height + 2 * self.pad[0]
        pad_width = width + 2 * self.pad[1]

        # initialiser dW et db avec le facteur de régularisation
        self.dW = self.reg * self.W
        self.db = self.reg * self.b

        dX = np.zeros((N, Fchannel, height, width))

        W_row = self.W.reshape(F, Fchannel * Fheight * Fwidth)

        dOut = self.activation['backward'](out) * dA

        for index in range(N):
            dOut_row = dOut[index].reshape(F, out_height * out_width)
            dX_col = W_row.T.dot(dOut_row)
            dX_block = np.zeros((Fchannel, pad_height, pad_width))

            col = 0
            for i in range(0, pad_height - Fheight + 1, self.stride[0]):
                for j in range(0, pad_width - Fwidth + 1, self.stride[1]):
                    dX_block[:, i:i+Fheight, j:j+Fwidth] += dX_col[:, col].reshape(Fchannel, Fheight, Fwidth)
                    col += 1

            if self.pad[0] > 0 and self.pad[1] > 0:
                dX[index] = dX_block[:, self.pad[0]:-self.pad[0], self.pad[1]:-self.pad[1]]
            elif self.pad[0] > 0:
                dX[index] = dX_block[:, self.pad[0]:-self.pad[0], :]
            elif self.pad[1] > 0:
                dX[index] = dX_block[:, :, self.pad[1]:-self.pad[1]]
            else:
                dX[index] = dX_block

            self.dW += dOut_row.dot(X_col[index].T).reshape(F, Fchannel, Fheight, Fwidth)
            self.db += dOut_row.sum(axis=1)

        return dX


class Conv2DCython(Conv2D):

    def forward(self, X, **kwargs):
        """Effectue la propagation avant cythonisée.

        Arguments:
            X {ndarray} -- Input de la couche. Shape (N, C, H, W)

        Returns:
            ndarray -- Scores de la couche
        """

        N, channel, height, width = X.shape
        F, Fchannel, Fheight, Fwidth = self.W.shape

        assert channel == Fchannel
        assert (height - Fheight + 2 * self.pad[0]) % self.stride[0] == 0
        assert (width - Fwidth + 2 * self.pad[1]) % self.stride[1] == 0

        out_height = np.uint32(1 + (height - Fheight + 2 * self.pad[0]) / self.stride[0])
        out_width = np.uint32(1 + (width - Fwidth + 2 * self.pad[1]) / self.stride[1])

        W_row = self.W.reshape(F, Fchannel*Fheight*Fwidth)

        X_col = np.asarray(im2col(X, N, channel, height, width,
                                  Fheight, Fwidth, 
                                  self.pad[0], self.pad[1], 
                                  self.stride[0], self.stride[1]))

        out = (W_row.dot(X_col) + self.b.reshape(F, 1))
        out = out.reshape(F, N, out_height, out_width).transpose(1, 0, 2, 3)

        self.cache = (X_col, out, height, width)

        A = self.activation['forward'](out)

        return A

    def backward(self, dA, **kwargs):
        """Effectue la rétropropagation cythonisée.

        Arguments:
            dA {ndarray} -- Dérivée de la loss par rapport au output de la couche.
                            Shape (N, F, out_height, out_width)

        Returns:
            ndarray -- Dérivée de la loss par rapport au input de la couche.
        """

        X_col, out, height, width = self.cache
        N, F, out_height, out_width = dA.shape
        _, Fchannel, Fheight, Fwidth = self.W.shape

        W_row = self.W.reshape(F, Fchannel * Fheight * Fwidth)

        dOut = self.activation['backward'](out) * dA
        dOut_mat = dOut.transpose(1, 0, 2, 3).reshape(F, N * out_height * out_width)

        self.dW = dOut_mat.dot(X_col.T).reshape(self.W.shape)
        self.dW += self.reg * self.W

        self.db = dOut_mat.sum(axis=1) 
        self.db += self.reg * self.b

        dX_col = W_row.T.dot(dOut_mat)
        dX = col2im(dX_col, N, Fchannel, height, width, Fheight, Fwidth, 
                    self.pad[0], self.pad[1], self.stride[0], self.stride[1])

        return np.asarray(dX)

    def reset(self):
        self.__init__(self.num_filters,
                      filter_size=self.filter_size,
                      channels=self.channels,
                      stride=self.stride,
                      padding=self.pad,
                      weight_scale=self.weight_scale,
                      activation=self.activation_id)

