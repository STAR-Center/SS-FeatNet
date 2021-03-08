import logging
import tensorflow as tf

from models.layers import conv2d
from models.layers import pairwise_dist
from models.pointnet_common import sample_points, sample_and_group, sample_and_group_all, query_and_group_points


def pointnet_sa_module(xyz, points, npoint, radius, nsample, mlp, mlp2, mlp3, is_training, scope, bn=True, bn_decay=None,
                       tnet_spec=None, knn=False, use_xyz=True,
                       keypoints=None, orientations=None, normalize_radius=True, final_relu=True):
    """ PointNet Set Abstraction (SA) Module. Modified to remove unneeded components (e.g. pooling),
        normalize points based on radius, and for a third layer of MLP

    Args:
        xyz (tf.Tensor): (batch_size, ndataset, 3) TF tensor
        points (tf.Tensor): (batch_size, ndataset, num_channel)
        npoint (int32): #points sampled in farthest point sampling
        radius (float): search radius in local region
        nsample (int): Maximum points in each local region
        mlp: list of int32 -- output size for MLP on each point
        mlp2: list of int32 -- output size for MLP after max pooling concat
        mlp3: list of int32 -- output size for MLP after second max pooling
        is_training (tf.placeholder): Indicate training/validation
        scope (str): name scope
        bn (bool): Whether to perform batch normalizaton
        bn_decay: Decay schedule for batch normalization
        tnet_spec: Unused in Feat3D-Net. Set to None
        knn: Unused in Feat3D-Net. Set to False
        use_xyz: Unused in Feat3D-Net. Set to True
        keypoints: If provided, cluster centers will be fixed to these points (npoint will be ignored)
        orientations (tf.Tensor): Containing orientations from the detector
        normalize_radius (bool): Whether to normalize coordinates [True] based on cluster radius.
        final_relu: Whether to use relu as the final activation function

    Returns:
        new_xyz: (batch_size, npoint, 3) TF tensor
        new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
        idx: (batch_size, npoint, nsample) int32 -- indices for local regions

    """

    with tf.variable_scope(scope) as sc:
        if npoint is None:
            nsample = xyz.get_shape()[1].value
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(xyz, points, use_xyz)
        else:
            new_xyz, new_points, idx, grouped_xyz, end_points = sample_and_group(npoint, radius, nsample, xyz, points, tnet_spec,
                                                                                 knn, use_xyz,
                                                                                 keypoints=keypoints,
                                                                                 orientations=orientations,
                                                                                 normalize_radius=normalize_radius)

        for i, num_out_channel in enumerate(mlp):
            new_points = conv2d(new_points, num_out_channel, kernel_size=[1, 1], stride=[1, 1], padding='VALID',
                                bn=bn, is_training=is_training,
                                scope='conv%d' % (i), reuse=False, )

        # Max pool
        pooled = tf.reduce_max(new_points, axis=[2], keep_dims=True)
        pooled_expand = tf.tile(pooled, [1, 1, new_points.shape[2], 1])

        # Concatenate
        new_points = tf.concat((new_points, pooled_expand), axis=3)

        if mlp2 is None: mlp2 = []
        for i, num_out_channel in enumerate(mlp2):
            new_points = conv2d(new_points, num_out_channel, [1, 1],
                                padding='VALID', stride=[1, 1],
                                bn=bn, is_training=is_training,
                                scope='conv_mid_%d' % (i), bn_decay=bn_decay,
                                activation=tf.nn.relu if (final_relu or i < len(mlp2) - 1) else None)

        # Max pool again
        new_points = tf.reduce_max(new_points, axis=[2], keep_dims=True)

        if mlp3 is None:
            mlp3 = []
        for i, num_out_channel in enumerate(mlp3):
            new_points = conv2d(new_points, num_out_channel, [1, 1],
                                padding='VALID', stride=[1, 1],
                                bn=bn, is_training=is_training,
                                scope='conv_post_%d' % (i), bn_decay=bn_decay,
                                activation=tf.nn.relu if (final_relu or i < len(mlp3) - 1) else None)
        new_points = tf.squeeze(new_points, [2])  # (batch_size, npoints, mlp2[-1])

        return new_xyz, new_points, idx, end_points


def feature_detection_module(xyz, points, num_clusters, radius, is_training, mlp, mlp2, num_samples=64, use_bn=True):
    """ Detect features in point cloud

    Args:
        xyz (tf.Tensor): Input point cloud of size (batch_size, ndataset, 3)
        points (tf.Tensor): Point features. Unused in 3DFeat-Net
        num_clusters (int): Number of clusters to extract. Set to -1 to use all points
        radius (float): Radius to consider for feature detection
        is_training (tf.placeholder): Set to True if training, False during evaluation
        mlp: list of int32 -- output size for MLP on each point
        mlp2: list of int32 -- output size for MLP on each region. Set to None or [] to ignore
        num_samples: Maximum number of points to consider per cluster
        use_bn: bool -- Whether to perform batch normalization

    Returns:
        new_xyz: Cluster centers
        idx: Indices of points sampled for the clusters
        attention: Output attention weights
        orientation: Output orientation (radians)
        end_points: Unused

    """
    end_points = {}
    new_xyz = sample_points(xyz, num_clusters)  # Sample point centers
    new_points, idx = query_and_group_points(xyz, points, new_xyz, num_samples, radius, knn=False, use_xyz=True,
                                             normalize_radius=True, orientations=None)  # Extract clusters

    # Pre pooling MLP
    for i, num_out_channel in enumerate(mlp):
        new_points = conv2d(new_points, num_out_channel, kernel_size=[1, 1], stride=[1, 1], padding='VALID',
                            bn=use_bn, is_training=is_training,
                            scope='conv%d' % (i), reuse=False, )

    # Max Pool
    new_points = tf.reduce_max(new_points, axis=[2], keep_dims=True)

    # Max pooling MLP
    if mlp2 is None: mlp2 = []
    for i, num_out_channel in enumerate(mlp2):
        new_points = conv2d(new_points, num_out_channel, [1, 1],
                            padding='VALID', stride=[1, 1],
                            bn=use_bn, is_training=is_training,
                            scope='conv_post_%d' % (i))

    # Attention and orientation regression
    with tf.variable_scope('attention') as sc:
        attention = conv2d(tf.stop_gradient(new_points), 128, [1, 1], stride=[1, 1], padding='VALID',
                       activation=tf.nn.relu, bn=False, reuse=False,scope='att1')
        attention = conv2d(attention, 128, [1, 1], stride=[1, 1], padding='VALID',
                       activation=tf.nn.relu, bn=False, reuse=False,scope='att2')
        attention = conv2d(attention, 1, [1, 1], stride=[1, 1], padding='VALID',
                       activation=tf.nn.softplus, bn=False, reuse=False,scope='att3')
    attention = tf.squeeze(attention, axis=[2, 3])

    orientation_xy = conv2d(new_points, 2, [1, 1], stride=[1, 1], padding='VALID',
                            activation=None, bn=False, scope='orientation', reuse=False)
    orientation_xy = tf.squeeze(orientation_xy, axis=2)
    orientation_xy = tf.nn.l2_normalize(orientation_xy, dim=2, epsilon=1e-8)
    orientation = tf.atan2(orientation_xy[:, :, 1], orientation_xy[:, :, 0])

    return new_xyz, idx, attention, orientation, end_points


def feature_extraction_module(l0_xyz, l0_points, is_training, mlp, mlp2, mlp3,
                              keypoints, orientations,
                              radius=2.0, num_samples=64, use_bn=True):
    """ Extract feature descriptors

    Args:
        l0_xyz (tf.Tensor): Input point cloud of size (batch_size, ndataset, 3)
        l0_points (tf.Tensor): Point features. Unused in 3DFeat-Net
        is_training (tf.placeholder): Set to True if training, False during evaluation
        mlp: list of int32 -- output size for MLP on each point
        mlp2: list of int32 -- output size for MLP after max pooling concat
        mlp3: list of int32 -- output size for MLP after second max pooling
        keypoints: Keypoints to compute features for
        orientations: Orientation (from detector) to pre-rotate clusters before compute descriptors
        radius: Radius to consider for feature detection
        num_samples: Maximum points in each local region
        use_bn: Whether to perform batch normalizaton

    Returns:
        xyz, features, end_points
    """

    # Extracts descriptors
    l1_xyz, l1_points, l1_idx, end_points = pointnet_sa_module(l0_xyz, l0_points, 512, radius, num_samples,
                                                   mlp=mlp, mlp2=mlp2, mlp3=mlp3,
                                                   is_training=is_training, scope='layer1',
                                                   bn=use_bn, bn_decay=None,
                                                   keypoints=keypoints, orientations=orientations, normalize_radius=True,
                                                   final_relu=False)

    xyz = l1_xyz
    features = tf.nn.l2_normalize(l1_points, dim=2, epsilon=1e-8)

    return xyz, features, end_points


class Feat3dNet:

    def __init__(self, param=None):
        """ Constructor: Sets the parameters for 3DFeat-Net

        Args:
            param:    Python dict containing the algorithm parameters. It should contain the
                      following fields (square brackets denote paper's parameters':
                      'NoRegress': Whether to skip regression of the keypoint orientation.
                                   [False] (i.e. regress)
                      'BaseScale': Cluster radius. [2.0] (as in the paper)
                      'Attention': Whether to predict the attention. [True]
                      'num_clusters': Number of clusters [512]
                      'num_samples': Maximum number of points per cluster [64]
                      'margin': Triplet loss margin [0.2]
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.param = {}
        self.param.update(param)
        self.logger.info('Model parameters: %s', self.param)

    def get_placeholders(self, data_dim):
        """ Gets placeholders for data, for triplet loss based training

        Args:
            data_dim: Dimension of point cloud. May be 3 (XYZ), 4 (XYZI), or 6 (XYZRGB or XYZNxNyNz)
                      However for Feat3D-Net we only use the first 3 values

        Returns:
            (anchor_pl, positive_pl, negative_pl)

        """
        anchor_pl = tf.placeholder(tf.float32, shape=(None, None, data_dim))  # type: tf.Tensor
        positive_pl = tf.placeholder(tf.float32, shape=(None, None, data_dim))  # type: tf.Tensor
        R_gt_pl = tf.placeholder(tf.float32, shape=(None,3,3))
        #negative_pl = tf.placeholder(tf.float32, shape=(None, None, data_dim))   # type: tf.Tensor
        return anchor_pl, positive_pl, R_gt_pl#, negative_pl

    def get_train_model(self, anchors, positives, is_training, use_bn=True):
        """ Constructs the training model. Essentially calls get_inference_model, but
            also handles the training triplets.

        Args:
            anchors (tf.Tensor): Anchor point clouds of size (batch_size, ndataset, 3).
            positives (tf.Tensor): Positive point clouds, same size as anchors
            negatives (tf.Tensor): Negative point clouds, same size as anchors
            is_training (tf.placeholder): Set to true only if training, false otherwise
            use_bn (bool): Whether to use batch normalization [True]

        Returns:
            xyz, features, anchor_attention, end_points

        """
        end_points = {}

        point_clouds = tf.concat([anchors, positives], axis=0)
        end_points['input_pointclouds'] = point_clouds

        xyz, features, attention, endpoints_temp = self.get_inference_model(point_clouds, is_training, use_bn)
        end_points['output_xyz'] = xyz
        end_points['output_features'] = features
        end_points.update(endpoints_temp)

        xyz = tf.split(xyz, 2, axis=0)
        features = tf.split(features, 2, axis=0)
        attentions = tf.split(attention, 2, axis=0)

        return xyz, features, attentions, end_points
    def tf_repeat(self, tensor, repeats):
        """
            https://github.com/tensorflow/tensorflow/issues/8246
            Args:

            input: A Tensor. 1-D or higher.
            repeats: A list. Number of repeat for each dimension, length must be the same as the number of dimensions in input

            Returns:

            A Tensor. Has the same type as input. Has the shape of tensor.shape * repeats
        """
        with tf.variable_scope("repeat"):
            expanded_tensor = tf.expand_dims(tensor, -1)
            multiples = [1] + repeats
            tiled_tensor = tf.tile(expanded_tensor, multiples = multiples)
            repeated_tesnor = tf.reshape(tiled_tensor, tf.shape(tensor) * repeats)
        return repeated_tesnor

    def get_inference_model(self, point_cloud, is_training, use_bn=True):
        """ Constructs the core 3DFeat-Net model.

        Args:
            point_cloud (tf.Tensor): Input point clouds of size (batch_size, ndataset, 3).
            is_training (tf.placeholder): Set to true only if training, false otherwise
            use_bn (bool): Whether to perform batch normalization

        Returns:
            xyz, features, attention, end_points

        """
        end_points = {}

        l0_xyz = point_cloud[:, :, :3]
        l0_points = None  # Normal information not used in 3DFeat-Net

        # Detection: Sample many clusters and computer attention weights and orientations
        num_clusters = self.param['num_clusters']
        with tf.variable_scope("detection") as sc:
            mlp = [64, 128, 256]
            mlp2 = [128, 64]
            keypoints, idx, attention, orientation, end_points_temp = \
                feature_detection_module(l0_xyz, l0_points, num_clusters, self.param['BaseScale'],
                                         is_training,
                                         mlp, mlp2, num_samples=self.param['num_samples'],
                                         use_bn=use_bn)

        end_points.update(end_points_temp)
        end_points['keypoints'] = keypoints
        end_points['attention'] = attention
        end_points['orientation'] = orientation

        keypoint_orientation = orientation

        if self.param['NoRegress']:
            keypoint_orientation = None
        if not self.param['Attention']:
            attention = None

        # Descriptor extraction: Extract descriptors for each cluster
        mlp = [32, 64]
        mlp2 = [128] if self.param['feature_dim'] <= 64 else [256]
        mlp3 = [self.param['feature_dim']]

        self.logger.info('Descriptor MLP sizes: {} | {} | {}'.format(mlp, mlp2, mlp3))
        with tf.variable_scope("description", reuse=tf.AUTO_REUSE) as sc:
            xyz, features, endpoints_temp = feature_extraction_module(l0_xyz, l0_points, is_training, mlp, mlp2, mlp3,
                                                                      keypoints=keypoints,
                                                                      orientations=keypoint_orientation,
                                                                      radius=self.param['BaseScale'],
                                                                      num_samples=self.param['num_samples'],
                                                                      use_bn=use_bn)
        end_points.update(endpoints_temp)

        return xyz, features, attention, end_points

    def get_loss(self, xyz, features, attentions, end_points, R_gt):
        """ Computes the attention weighted alignment loss as described in our paper.

        Args:
            xyz: Keypoint coordinates (Unused)
            features: List of [anchor_features, positive_features, negative_features]
            anchor_attention: Attention from anchor point clouds
            end_points: end_points, which will be augmented and returned

        Returns:
            loss, end_points
        """

        X, Y = xyz
        anchors, positives = features
        anch_atten, posi_atten = attentions
        anch_atten /= tf.reduce_sum(anch_atten,axis=1,keepdims=True)
        posi_atten /= tf.reduce_sum(posi_atten,axis=1,keepdims=True)
        #anchors = tf.multiply(anchors, tf.expand_dims(anch_atten,2))
        #positives = tf.multiply(positives, tf.expand_dims(posi_atten,2))

        # Computes for each feature of the anchor, the distance to the nearest feature in the positive and negative
        with tf.variable_scope("alignment") as sc:
            positive_dist = pairwise_dist(tf.stop_gradient(anchors), tf.stop_gradient(positives))
            w = tf.reshape(tf.exp(-positive_dist), [tf.shape(positive_dist)[0], -1])
            self.w = tf.exp(-positive_dist)

            # attention weight
            self.w = tf.multiply(tf.expand_dims(anch_atten,2),self.w)
            self.w = tf.multiply(self.w, tf.expand_dims(posi_atten,1))

            w = tf.reshape(self.w, [tf.shape(positive_dist)[0],-1])

            # solving

            w3 = tf.expand_dims(w,1)# for computing
            w3 = tf.tile(w3,[1,3,1])

            w_sum = tf.reduce_sum(w,axis=1)

            w_sum3 = tf.expand_dims(w_sum,1)
            w_sum3 = tf.tile(w_sum3,[1,3])
            # np.repeat(X, Y.shape[1], axis=1)
            # follow https://github.com/tensorflow/tensorflow/issues/8246
            X_ = self.tf_repeat(X, [1, tf.shape(Y)[1], 1]) 

            # np.tile(Y, (1, X.shape[0], 1))
            Y_ = tf.tile(Y, [1, tf.shape(X)[1],1])
            
            # shift
            WY = tf.transpose(tf.transpose(Y_,[0,2,1])*w3, [0,2,1])
            WX = tf.transpose(tf.transpose(X_,[0,2,1])*w3, [0,2,1])
            Ycenter = tf.reduce_sum(WY,axis=1)/w_sum3
            Xcenter = tf.reduce_sum(WX,axis=1)/w_sum3

            Ycenter3 = tf.expand_dims(Ycenter,1)
            Ycenter3 = tf.tile(Ycenter3,[1,tf.shape(Y_)[1],1])
            Xcenter3 = tf.expand_dims(Xcenter,1)
            Xcenter3 = tf.tile(Xcenter3,[1,tf.shape(X_)[1],1])

            Y_ -= Ycenter3
            X_ -= Xcenter3

            # solve
            WY_T = tf.transpose(tf.multiply(tf.transpose(Y_,[0,2,1]),w3), [0,2,1])
            S = tf.matmul(tf.transpose(X_,[0,2,1]),WY_T)
            self.S = S

            s, u, v = tf.linalg.svd(S, full_matrices=True, compute_uv=True)
            self.u = u
            self.v = v
            

            '''
            cond = tf.linalg.det(v) * tf.linalg.det(u) < 0
            def true_func(vx):
                mask = tf.constant([[1,1,-1],[1,1,-1],[1,1,-1]], tf.float32)
                mask = tf.expand_dims(mask,0)
                mask = tf.tile(mask, [tf.shape(vx)[0],1,1])
                return vx*mask

            v = tf.where(cond, true_func(v), v)
            '''
            

            R_pred = tf.matmul(v,u, adjoint_b=True)#tf.transpose(u, [0,2,1]))
            self.R_pred = R_pred

            #negative_dist = pairwise_dist(anchors, negatives)
            #best_positive = tf.reduce_min(positive_dist, axis=2)
            #best_negative = tf.reduce_min(negative_dist, axis=2)

        with tf.variable_scope("triplet_loss") as sc:
            if True:
                eye = tf.eye(3)
                eye = tf.expand_dims(eye,0)
                eye = tf.tile(eye,[tf.shape(R_pred)[0],1,1])
                diff = tf.squared_difference(eye, tf.matmul(R_pred, tf.transpose(R_gt,[0,2,1])))
                #sum_positive = tf.reduce_mean(best_positive, 1)
                #sum_negative = tf.reduce_mean(best_negative, 1)
            else:
                attention_sm = anchor_attention / tf.reduce_sum(anchor_attention, axis=1)[:, None]
                sum_positive = tf.reduce_sum(attention_sm * best_positive, 1)
                sum_negative = tf.reduce_sum(attention_sm * best_negative, 1)

                tf.summary.histogram('normalized_attention', attention_sm)
                end_points['normalized_attention'] = attention_sm

            #end_points['sum_positive'] = sum_positive
            #end_points['sum_negative'] = sum_negative
            #triplet_cost = tf.maximum(0., sum_positive - sum_negative + self.param['margin'])

            #loss = tf.reduce_mean(triplet_cost)
            loss = tf.reduce_sum(tf.reduce_sum(diff, axis=2), axis=1)
            loss = tf.reduce_mean(loss)

        tf.summary.scalar('loss', loss)

        return loss, end_points

    def get_train_op(self, loss_op, lr=1e-5, global_step=None):
        """ Gets training op
        """

        optimizer = tf.train.AdamOptimizer(lr)

        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        to_exclude = []
        if self.param['freeze_scopes'] is not None:
            for s in self.param['freeze_scopes']:
                to_exclude += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=s)
        var_list = [v for v in var_list if v not in to_exclude]

        train_op = optimizer.minimize(loss_op, global_step=global_step,
                                      var_list=var_list)
        return train_op

