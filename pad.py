import math


def padding_mode_zeros_L_out(kernel_size, input_size, padding, stride, dilation):
    """

        This function computes the padding of a single mode (of a tensor, which is a vector)
        when the padding_mode is 'zeros'.

        Inputs:

        TODO: ADD the input parameters, quite strict forward

        Output:

        [padding] (int) - zero-padding added to both sides of the input (the single mode)

        Padding is calculated using the formula below

        ref: https://pytorch.org/docs/master/generated/torch.nn.Conv1d.html

        TODO: Since in Conv1d, the padding can be computed by default, check the consistency.

    """



    return math.floor(input_size + (2*padding - dilation*(kernel_size-1) - 1)/stride + 1)


###
# This computes the zero padding to append to the input image vector so that
# the resulting convolution has the same length as the maximum length of the kernel
# and the input tensor. Note the kernel can be longer than the input tensor. Additionally,
# conv_einsum("i, i -> i | i", A, B, padding=) = conv_einsum("i, i -> i | i", B, A, padding=)
# when this padding is selecting. These two properties may not hold when dilation > 1.
# \todo
#
def max_zeros_padding_1d(ker_mode_size, input_mode_size, \
                         max_mode_size, stride=1, dilation=1):
    """

        TODO: READ the comment below and solve it.

    """
    # see https://pytorch.org/docs/master/generated/torch.nn.Conv1d.html for the definition
    # under the Shape section for the definition of the padding. We add 1 so the integer division
    # by 2 errs large instead of small
    # for 1d convolutions ker_mode_size < input_mode_size but this is not true for higher
    # dimensions, and I use this function to compute those paddings

    # "input" tensor is synonymous with "image" tensor in much of this

    max_ker_input = max(ker_mode_size, input_mode_size, max_mode_size)

    # \todo it's not clear if this works if stride doesn't divide evenly into
    #           input_mode_size + 2*padding - dilation * (kernel_size - 1) - 1
    #       (see Conv1d documentation)
    #       It does however appear to work whenever dilation = 1, or when
    #       dilation*kernel_size < input_size + some factor involving stride
    # padding can only be negative if kernel_mode_size == 0
    twice_padding = (max_ker_input-1)*stride + 1 + dilation*(ker_mode_size-1) - input_mode_size

    return (twice_padding+1)//2 # add 1 so its never less than twice_padding/2




def max_zeros_padding_nd(ker_mode_size, input_mode_size, \
                         max_mode_size, stride=1, dilation=1):
    return tuple(max_zeros_padding_1d(ker_mode_size[i], input_mode_size[i], \
                                   max_mode_size[i], stride[i], dilation[i]) \
                 for i in range(0,len(ker_mode_size)))


def pad(input_size, kernel_size, padding_mode, max_mode_sizes, stride, dilation):

    """
    This functions computes the padding given a padding mode.

    Inputs:

    [input_size] () - The size of the input tensor

    [kernel_size] () - The size of the kernel tensor

    [max_mode_sizes] (tuple)

    [stride] (tuple)

    [dilation] (tuple)

    TODO: figure out the type of the input_size (possibly a list)
    TODO: figure out the type of the kernel_size (possibly a list)

    [padding_mode] (string, optional) –
                'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'

                Implemented: zeros

                Not Implemented: 'reflect', 'replicate','circular'


    Output:

    [padding] (tuple) –
                Zero-padding added to both sides of the input. Default: 0



    """

    if padding_mode == "zeros":
        padding = max_zeros_padding_nd(kernel_size, input_size, \
                                       max_mode_size=max_mode_sizes, \
                                       stride=stride, dilation=dilation)

    #TODO: figuring out what is the max_zeros
    else:
         raise NotImplementedError("The intended padding mode is not implemented.")

    return padding
