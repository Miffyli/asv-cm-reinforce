# models.py
# PyTorch models for the ASV-CM optimization

import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseWithDiscriminator(nn.Module):
    """
    Siamese network with a discriminator block.

    Essentially: Run two inputs through same network, combine and continue to
                 some arbritrary output.
    Designed for the mutual information learning:
        https://arxiv.org/abs/1812.00271
    Returns a linear output.
    """
    def __init__(
        self,
        feature_size,
        output_size,
        feature_network,
        num_discriminator_layers,
        num_discriminator_units,
        discriminator_dropout=0.0,
    ):
        """
        feature_size: Size of the 1D vectors from feature network
        feature_network: nn.Module to be used for processing inputs.
        num_discriminator_layers/units: Number of layers after combining
                                        the two streams.
        discriminator_dropout: Probability of dropping nodes in discriminator
                               layers
        """
        super().__init__()
        self.output_size = output_size
        self.feature_size = feature_size
        self.num_discriminator_units = num_discriminator_units
        self.num_discriminator_layers = num_discriminator_layers
        self.discriminator_dropout = discriminator_dropout

        self.feature_network = feature_network

        last_num_units = feature_size * 2
        self.discriminator_layers = []
        for i in range(num_discriminator_layers):
            self.discriminator_layers.append(nn.Linear(last_num_units, num_discriminator_units))

            # Add dropout if enabled
            if discriminator_dropout is not None:
                self.discriminator_layers.append(nn.Dropout(p=discriminator_dropout))

            self.discriminator_layers.append(nn.ReLU())
            last_num_units = num_discriminator_units
        # Add final, linear output

        self.discriminator_layers.append(
            nn.Linear(last_num_units, output_size)
        )
        self.discriminator = nn.Sequential(*self.discriminator_layers)

        # If these are set to some torch.Parameters,
        # we will feed these encodings to the discriminator
        # rather than the inputs we get
        self.fixed_x1_encoding = None
        self.fixed_x2_encoding = None

    def forward(self, x1, x2, direct_x1=None, direct_x2=None):
        """
        Run two samples through the network

        If direct_x1/direct_x2 is given, then x1/x2 features will
        be ignored and direct version is fed directly to discriminator.

        If either fixed_x1_encoding or fixed_x2_encoding is set to
        some PyTorch tensor, that is fed to the discriminator rather
        than any of the inputs
        """
        # Run the two samples through
        # shared parameters, unless we have
        # fixed encodings
        if self.fixed_x1_encoding is not None:
            # Sanity check
            assert x1 is None and direct_x1 is None, "x1 input was fixed"
            x1 = self.fixed_x1_encoding
        elif direct_x1 is not None:
            assert x1 is None, "x1 received both encoded and raw input"
            x1 = direct_x1
        else:
            x1 = self.feature_network(x1)

        if self.fixed_x2_encoding is not None:
            # Sanity check
            assert x2 is None and direct_x2 is None, "x2 input was fixed"
            x2 = self.fixed_x2_encoding
        if direct_x2 is not None:
            assert x2 is None, "x2 received both encoded and raw input"
            x2 = direct_x2
        else:
            x2 = self.feature_network(x2)

        # Combine the results
        x = torch.cat((x1, x2), dim=1)

        # Run results through the discriminator block
        output = self.discriminator(x)

        return output


class TimeDilatedConv(nn.Module):
    """
    Perform a operation similar to Time Dilated NN layers in:
        https://www.danielpovey.com/files/2015_interspeech_multisplice.pdf

    Inputs: 1xNxD tensors
    Outputs: 1xNxD tensors
    """

    def __init__(
        self,
        feature_size,
        context_size,
        subsample=False
    ):
        """
        subsample (bool): If True, use subsampling technique instead of
                          the full context (i.e. {-2, 2} rather than
                          [-2, 2], using the notation used in the paper
                          linked in the class comments).
        """
        super().__init__()
        self.feature_size = feature_size
        self.context_size = context_size
        self.subsample = subsample


        if subsample:
            # Only take the element at the edges of the context window.
            # Do this by having a convolution that covers two vectors and
            # using dilation to create the gap between them
            self.cnn_op = nn.Conv2d(1, feature_size, (2, feature_size),
                                    dilation=(context_size * 2, 1))
        else:
            # Full context.
            # Context size goes both ways, and we also include
            # the current sample (around which the filter is centered)
            kernel_width = context_size * 2 + 1
            self.cnn_op = nn.Conv2d(1, feature_size, (kernel_width, feature_size))

    def forward(self, x):
        x = self.cnn_op(x)
        # Move the filter (channel) dimension to the feature_size
        # channel
        x = torch.transpose(x, 1, 3)
        return x


class PoolStatistics(nn.Module):
    """
    Pool statistics over a 2D matrix, replacing that matrix with
    1D vector of pooled statistics
    Inputs: NxM tensors
    Outputs: K tensors, where X depends on number of statistics included
             and dimension over which reduction is done
    """
    def __init__(
        self,
        dim,
        include_mean,
        include_std,
        include_max
    ):
        super().__init__()
        self.dim = dim
        self.include_mean = include_mean
        self.include_std = include_std
        self.include_max = include_max

    def forward(self, x):
        vectors_to_concat = []

        if self.include_mean:
            means = torch.mean(x, dim=self.dim)
            vectors_to_concat.append(means)
        if self.include_std:
            stds = torch.std(x, dim=self.dim)
            vectors_to_concat.append(stds)
        if self.include_max:
            maxs = torch.max(x, dim=self.dim)[0]
            vectors_to_concat.append(maxs)

        x = torch.cat(vectors_to_concat, dim=1)
        return x


class XVectorLikeNetwork(nn.Module):
    """
    A network similar used to extract X-vectors, described here:
        https://danielpovey.com/files/2017_interspeech_embeddings.pdf
    Main complexity comes from the Time-Dilated NN layers:
        https://www.danielpovey.com/files/2015_interspeech_multisplice.pdf

    In short: Process feature matrices (NxD) of varying length (N) by
              first running it over TDNN layers (second reference), pool
              statistics over frames for a fixed sized array and process
              them bit further

    NOTE: TDNN allows mapping the nearby context to new number of units, rather
          than being stuck in the original feature size. Current implementation
          here is stuck with output size that matches the input feature size.
    """
    def __init__(
        self,
        feature_size,
        output_size,
        include_mean=True,
        include_std=False,
        include_max=False,
        dropout=0.0
    ):
        """
        feature_size: Number of features (dimensionality) per frame
        output_size: Number of units in final output
        include*: Different statistics to include in the statistics pooling
        """
        super().__init__()
        self.feature_size = feature_size
        self.output_size = output_size
        self.include_mean = include_mean
        self.include_std = include_std
        self.include_max = include_max
        self.dropout = dropout

        # Context sizes in order: [-2, 2], {-2, 2}, {-3, 3}
        # A little trickery here: We have as many filters/feature-maps
        # as we want outputs, and then transpose that dimension on
        # y-axis.
        # Context size == 0 means we do not use nearby features,
        # essentially turning the operation into a simple 1D conv with
        # only one filter.
        self.td1 = TimeDilatedConv(feature_size, 2)
        self.td2 = TimeDilatedConv(feature_size, 2, subsample=True)
        self.td3 = TimeDilatedConv(feature_size, 3, subsample=True)
        self.td4 = TimeDilatedConv(feature_size, 0)
        self.td5 = TimeDilatedConv(feature_size, 0)
        # Combine all feature-level operations into one operation
        self.feature_level_operations = nn.Sequential(
            self.td1,
            nn.ReLU(),
            self.td2,
            nn.ReLU(),
            self.td3,
            nn.ReLU(),
            self.td4,
            nn.ReLU(),
            self.td5,
        )

        self.statistics_pooling_operation = PoolStatistics(
            dim=1,
            include_mean=include_mean,
            include_std=include_std,
            include_max=include_max
        )
        # Due to our restrictions we know that there will be feature_size number of
        # values per statistic
        stats_pool_output_size = (include_mean + include_std + include_max) * feature_size
        # Segment-level operations: Just yer standard denses/FC/linears
        # Note: 256 units per layer vs. 512 of the original network
        self.segment_level_operations = nn.Sequential(
            nn.Linear(stats_pool_output_size, 256),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(256, output_size)
        )

    def forward(self, x_list):
        """
        Process A LIST of tensors. Here the length of the takes place of the
        batch dimension, as we can not put all features of different size into
        one tensor without a lot of padding
        """

        # Process each item separately (Uhf...) to fixed sized tensors,
        # batch together and run through final layers together
        x_items = []
        for x in x_list:
            # Add batch dim and channel dim to the individual
            # operations
            x = x[None, None]
            x = self.feature_level_operations(x)
            # Remove channel dim
            x = x[:, 0]
            x = self.statistics_pooling_operation(x)
            x_items.append(x)
        x = torch.cat(x_items, dim=0)

        x = self.segment_level_operations(x)

        return x


class ConvolutionFeatureProcessor(nn.Module):
    """
    Simpler varying-length feature processor network using
    2D convolutions and statistics pooling. Essentially
    X-vector network but smaller.
    """
    def __init__(
        self,
        feature_size,
        output_size,
        include_mean=True,
        include_std=False,
        include_max=False
    ):
        """
        feature_size: Number of features (dimensionality) per frame
        output_size: Number of units in final output
        include*: Different statistics to include in the statistics pooling
        """
        super().__init__()
        self.feature_size = feature_size
        self.output_size = output_size
        self.include_mean = include_mean
        self.include_std = include_std
        self.include_max = include_max

        # Two convolutions, one with context size 1 (one left and right),
        # and then one to just post-process features a little without
        # context
        self.td1 = TimeDilatedConv(feature_size, 1)
        self.td2 = TimeDilatedConv(feature_size, 0)
        # Combine all feature-level operations into one operation
        self.feature_level_operations = nn.Sequential(
            self.td1,
            nn.ReLU(),
            self.td2,
        )

        self.statistics_pooling_operation = PoolStatistics(
            dim=1,
            include_mean=include_mean,
            include_std=include_std,
            include_max=include_max
        )
        # Due to our restrictions we know that there will be feature_size number of
        # values per statistic
        stats_pool_output_size = (include_mean + include_std + include_max) * feature_size
        # Segment-level operations: Just yer standard denses/FC/linears
        # Note: 256 units per layer vs. 512 of the original network
        self.segment_level_operations = nn.Sequential(
            nn.Linear(stats_pool_output_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        )

    def forward(self, x_list):
        """
        Process A LIST of tensors. Here the length of the takes place of the
        batch dimension, as we can not put all features of different size into
        one tensor without a lot of padding
        """

        # Process each item separately (Uhf...) to fixed sized tensors,
        # batch together and run through final layers together
        x_items = []
        for x in x_list:
            # Add batch dim and channel dim to the individual
            # operations
            x = x[None, None]
            x = self.feature_level_operations(x)
            # Remove channel dim
            x = x[:, 0]
            x = self.statistics_pooling_operation(x)
            x_items.append(x)
        x = torch.cat(x_items, dim=0)

        x = self.segment_level_operations(x)

        return x
