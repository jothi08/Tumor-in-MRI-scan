import tensorflow as tf

#https://stackoverflow.com/questions/62259112/dice-loss-becomes-nan-after-some-epochs
def resblock_in_resnet(X,n_filters,stride):
    # check if the number of n_filters needs to be increase, assumes channels last format ->One solution is to use a 1×1 convolution layer
    #print("see:",X.shape[-1])
    #if X.shape[-1] != n_filters:
    #  X = tf.keras.layers.Conv2D(n_filters,padding='same', kernel_size = (1,1))(X)
    if stride=="No":
        X = tf.keras.layers.Conv2D(n_filters,padding='same', kernel_size = (3,3))(X)
    else:
        X = tf.keras.layers.Conv2D(n_filters,padding='same', kernel_size = (3,3),strides=(2,2))(X)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Activation("relu")(X)
    #X = tf.keras.layers.ZeroPadding2D(padding=1)(X)
    X = tf.keras.layers.Conv2D(n_filters,padding='same', kernel_size = (3,3))(X)
    #getting nan for dice loss and dice coff . Hence adding below batchnormalization layer after every convolution layer
    #X = tf.keras.layers.BatchNormalization()(X)

    return X
    
def decoder_block(X,n_filters):

    X = tf.keras.layers.Conv2D(n_filters,padding='same', kernel_size = (3,3))(X)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Activation("relu")(X)
    #getting nan for dice loss and dice coff . Hence adding below batchnormalization after every convolution layer

    X = tf.keras.layers.Conv2D(n_filters,padding='same', kernel_size = (3,3))(X)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Activation("relu")(X)
    #getting nan for dice loss and dice coff . Hence adding below batchnormalization after every convolution layer
    #X = tf.keras.layers.BatchNormalization()(X)

    return X
    
def batch_norm_and_activation(X):
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Activation("relu")(X)
    #X = tf.keras.layers.ZeroPadding2D(padding=1)(X)
    return X
  
def up_sampling(X,merge_input):
    """
    Upsampling and concatination with the side path (merge_input)
    """
    X = tf.keras.layers.UpSampling2D((2,2))(X)
    concat = tf.keras.layers.Concatenate()([X,merge_input])
    return concat
    

#input_shape = [256,256,3]
def built_unet_resnet34(input_shape):
    input = tf.keras.Input(input_shape)

    BN1 = tf.keras.layers.BatchNormalization()(input)
    conv_1 = tf.keras.layers.Conv2D(64,padding='same', activation='relu', kernel_size = (3,3), strides=(2,2) ,name="conv_1")(BN1)
    conv_1 = tf.keras.layers.BatchNormalization()(conv_1)

    activ1 = tf.keras.layers.Activation("relu")(conv_1)
    pool_1 = tf.keras.layers.MaxPool2D((2,2))(activ1)

    init_bna = batch_norm_and_activation(pool_1)
    conv_2 = tf.keras.layers.Conv2D(64,padding='same', kernel_size = (3,3))(init_bna)
    resblock1 = resblock_in_resnet(init_bna,64,"No")
    add_1 = tf.keras.layers.Add()([resblock1, conv_2])

    add_1_bna = batch_norm_and_activation(add_1)
    resblock2 = resblock_in_resnet(add_1_bna,64,"No")
    add_2 = tf.keras.layers.Add()([resblock2, add_1])

    add_2_bna = batch_norm_and_activation(add_2)
    resblock3 = resblock_in_resnet(add_2_bna,64,"No")
    add_3 = tf.keras.layers.Add()([resblock3, add_2])
    add_3_conv2d=tf.keras.layers.Conv2D(128,padding='same', activation='relu', kernel_size = (3,3),strides=(2,2))(add_3)

    add_3_bna = batch_norm_and_activation(add_3)
    resblock4 = resblock_in_resnet(add_3_bna,128,"Yes")
    add_4 = tf.keras.layers.Add()([resblock4, add_3_conv2d])

    add_4_bna = batch_norm_and_activation(add_4)
    resblock5 = resblock_in_resnet(add_4_bna,128,"No")
    add_5 = tf.keras.layers.Add()([resblock5, add_4])


    add_5_bna = batch_norm_and_activation(add_5)
    resblock6 = resblock_in_resnet(add_5_bna,128,"No")
    add_6 = tf.keras.layers.Add()([resblock6, add_5])


    add_6_bna = batch_norm_and_activation(add_6)
    resblock7 = resblock_in_resnet(add_6_bna,128,"No")
    add_7 = tf.keras.layers.Add()([resblock7, add_6])

    add_7_conv2d=tf.keras.layers.Conv2D(256,padding='same', activation='relu', kernel_size = (3,3),strides=(2,2))(add_7)

    add_7_bna = batch_norm_and_activation(add_7)
    resblock8 = resblock_in_resnet(add_7_bna,256,"Yes")
    add_8 = tf.keras.layers.Add()([resblock8, add_7_conv2d])

    add_8_bna = batch_norm_and_activation(add_8)
    resblock9 = resblock_in_resnet(add_8_bna,256,"No")
    add_9 = tf.keras.layers.Add()([resblock9, add_8])


    add_9_bna = batch_norm_and_activation(add_9)
    resblock10 = resblock_in_resnet(add_9_bna,256,"No")
    add_10 = tf.keras.layers.Add()([resblock10, add_9])


    add_10_bna = batch_norm_and_activation(add_10)
    resblock11 = resblock_in_resnet(add_10_bna,256,"No")
    add_11 = tf.keras.layers.Add()([resblock11, add_10])


    add_11_bna = batch_norm_and_activation(add_11)
    resblock12 = resblock_in_resnet(add_11_bna,256,"No")
    add_12 = tf.keras.layers.Add()([resblock12, add_11])

    add_12_bna = batch_norm_and_activation(add_12)
    resblock13 = resblock_in_resnet(add_12_bna,256,"No")
    add_13 = tf.keras.layers.Add()([resblock13, add_12])

    add_13_conv2d=tf.keras.layers.Conv2D(512,padding='same', activation='relu', kernel_size = (3,3),strides=(2,2))(add_13)


    add_13_bna = batch_norm_and_activation(add_13)
    resblock14 = resblock_in_resnet(add_13_bna,512,"Yes")
    add_14 = tf.keras.layers.Add()([resblock14, add_13_conv2d])

    add_14_bna = batch_norm_and_activation(add_14)
    resblock15 = resblock_in_resnet(add_14_bna,512,"No")
    add_15 = tf.keras.layers.Add()([resblock15, add_14])


    add_15_bna = batch_norm_and_activation(add_15)
    resblock16 = resblock_in_resnet(add_15_bna,512,"No")
    add_16 = tf.keras.layers.Add()([resblock16, add_15])

    add_16_bna = batch_norm_and_activation(add_16)

    #decoder_stage ->ds
    ds_upsample_0=up_sampling(add_16_bna,add_12_bna)
    ds_0=decoder_block(ds_upsample_0,256)
    ds_upsample_1=up_sampling(ds_0,add_6_bna)
    ds_1=decoder_block(ds_upsample_1,128)
    ds_upsample_2=up_sampling(ds_1,add_2_bna)
    ds_2=decoder_block(ds_upsample_2,64)
    ds_upsample_3=up_sampling(ds_2,activ1)
    ds_3=decoder_block(ds_upsample_3,64)
    upsample1 = tf.keras.layers.UpSampling2D((2,2))(ds_3)
    ds_4=decoder_block(upsample1,64)
    final_conv = tf.keras.layers.Conv2D(1,padding='same',kernel_size = (1,1),name="final_conv")(ds_4)
    output = tf.keras.layers.Activation("sigmoid")(final_conv)
    unet_with_resnet34 =tf.keras.Model(input,output)
    return unet_with_resnet34