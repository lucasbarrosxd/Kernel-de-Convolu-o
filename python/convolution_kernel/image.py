"""Define funções de processamento de imagem que utilizam o kernel de convolução."""
# Importações # ------------------------------------------------------------------------------------------------------ #
# Módulos
from PIL import Image
# Locais
from .convolution_kernel import ConvolutionKernel


# Funções # ---------------------------------------------------------------------------------------------------------- #
def edge_detection(image: Image.Image) -> Image.Image:
    # TODO - edge_detection: docstring desta função.
    kernel = ConvolutionKernel(
        matrix=[[0, -1, 0], [-1, 0, 1], [0, 1, 0]],
        anchor=(1, 1)
    )

    brightness = [
        [
            sum(image.getpixel((x_index, y_index))[:3]) / 3
            for y_index in range(image.size[1])
        ]
        for x_index in range(image.size[0])
    ]

    out_brightness = kernel.apply(
        function=lambda coord: brightness[coord[0]][coord[1]],
        limits=image.size,
        weight=1,
        default=0
    )

    out_image = Image.new(mode="RGBA", size=image.size)

    for pixel_iy in range(out_image.size[1]):
        for pixel_ix in range(out_image.size[0]):
            out_image.putpixel(
                xy=(pixel_ix, pixel_iy),
                value=(
                    int(out_brightness[pixel_ix][pixel_iy]),
                    int(out_brightness[pixel_ix][pixel_iy]),
                    int(out_brightness[pixel_ix][pixel_iy]),
                    255
                )
            )

    return out_image


def box_blur(image: Image.Image) -> Image.Image:
    # TODO - box_blur: a docstring desta função.
    kernel = ConvolutionKernel(
        matrix=[[0, 1, 0], [1, 0, 1], [0, 1, 0]],
        anchor=(1, 1)
    )

    out_r = kernel.apply(function=lambda coord: image.getpixel(coord)[0], limits=image.size, weight=4, default=0)
    out_g = kernel.apply(function=lambda coord: image.getpixel(coord)[1], limits=image.size, weight=4, default=0)
    out_b = kernel.apply(function=lambda coord: image.getpixel(coord)[2], limits=image.size, weight=4, default=0)
    out_a = kernel.apply(function=lambda coord: image.getpixel(coord)[3], limits=image.size, weight=4, default=255)

    out_image = Image.new(mode="RGBA", size=image.size)

    for pixel_iy in range(out_image.size[1]):
        for pixel_ix in range(out_image.size[0]):
            out_image.putpixel(
                xy=(pixel_ix, pixel_iy),
                value=(
                    int(out_r[pixel_ix][pixel_iy]),
                    int(out_g[pixel_ix][pixel_iy]),
                    int(out_b[pixel_ix][pixel_iy]),
                    int(out_a[pixel_ix][pixel_iy])
                )
            )

    return out_image


def gaussian_blur(image: Image.Image) -> Image.Image:
    # TODO - gaussian_blur: a docstring desta função.
    kernel = ConvolutionKernel(
        matrix=[[1, 2, 1], [2, 4, 2], [1, 2, 1]],
        anchor=(1, 1)
    )

    out_r = kernel.apply(function=lambda coord: image.getpixel(coord)[0], limits=image.size, weight=16, default=0)
    out_g = kernel.apply(function=lambda coord: image.getpixel(coord)[1], limits=image.size, weight=16, default=0)
    out_b = kernel.apply(function=lambda coord: image.getpixel(coord)[2], limits=image.size, weight=16, default=0)
    out_a = kernel.apply(function=lambda coord: image.getpixel(coord)[3], limits=image.size, weight=16, default=255)

    out_image = Image.new(mode="RGBA", size=image.size)

    for pixel_iy in range(out_image.size[1]):
        for pixel_ix in range(out_image.size[0]):
            out_image.putpixel(
                xy=(pixel_ix, pixel_iy),
                value=(
                    int(out_r[pixel_ix][pixel_iy]),
                    int(out_g[pixel_ix][pixel_iy]),
                    int(out_b[pixel_ix][pixel_iy]),
                    int(out_a[pixel_ix][pixel_iy])
                )
            )

    return out_image


def sharpen(image: Image.Image) -> Image.Image:
    # TODO - sharpen: a docstring desta função.
    kernel = ConvolutionKernel(
        matrix=[[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
        anchor=(1, 1)
    )

    out_r = kernel.apply(function=lambda coord: image.getpixel(coord)[0], limits=image.size, weight=1, default=0)
    out_g = kernel.apply(function=lambda coord: image.getpixel(coord)[1], limits=image.size, weight=1, default=0)
    out_b = kernel.apply(function=lambda coord: image.getpixel(coord)[2], limits=image.size, weight=1, default=0)
    out_a = kernel.apply(function=lambda coord: image.getpixel(coord)[3], limits=image.size, weight=1, default=255)

    out_image = Image.new(mode="RGBA", size=image.size)

    for pixel_iy in range(out_image.size[1]):
        for pixel_ix in range(out_image.size[0]):
            out_image.putpixel(
                xy=(pixel_ix, pixel_iy),
                value=(
                    int(out_r[pixel_ix][pixel_iy]),
                    int(out_g[pixel_ix][pixel_iy]),
                    int(out_b[pixel_ix][pixel_iy]),
                    int(out_a[pixel_ix][pixel_iy])
                )
            )

    return out_image


def embossing(image: Image.Image) -> Image.Image:
    # TODO - embossing: a docstring desta função.
    # TODO - embossing: implementar esta função.
    raise NotImplementedError
