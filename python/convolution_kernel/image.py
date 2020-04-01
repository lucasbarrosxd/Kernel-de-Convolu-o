"""Define funções de processamento de imagem que utilizam o kernel de convolução."""
# Importações # ------------------------------------------------------------------------------------------------------ #
# Módulos
from PIL import Image
# Locais
from .convolution_kernel import ConvolutionKernel


# Funções # ---------------------------------------------------------------------------------------------------------- #
def edge_detection(image: Image.Image) -> Image.Image:
    """Cria uma nova imagem com de arestas detectadas na imagem passada.

    O algoritmo de detecção utiliza o kernel de convolução passado no vídeo relacionado à tarefa.

    Parâmetros
    ----------
    image : Image.Image
        A imagem a partir da qual a nova imagem de arestas detectadas será gerada.

        A imagem deve ser uma imagem da biblioteca PIL (ou Pillow), e deve ter formato RGB ou RGBA [err #1].

    Retorno
    -------
    Image.Image
        A imagem de arestas detectadas, gerada a partir da imagem passada. É uma imagem da biblioteca PIL (ou Pillow)
          em formato RGBA.

    Erros
    -----
    ValueError
    [1] Caso a imagem passada esteja em um formato que não seja RGB ou RGBA.
    """
    # Verificar se o formato da imagem está correto.
    if image.mode != "RGB" and image.mode != "RGBA":
        raise ValueError("[1] O formato da imagem deve ser RGB ou RGBA.")

    # Construir o kernel de convolução.
    kernel = ConvolutionKernel(
        matrix=[[0, -1, 0], [-1, 0, 1], [0, 1, 0]],
        anchor=(1, 1)
    )

    # Construir matriz do brilho de cada pixel previamente, pois cada pixel será chamado múltiplas vezes, e calcular a
    #   média múltiplas vezes deixaria o código mais pesado. É considerado que o brilho é a média dos valores RGB.
    brightness = [
        [
            sum(image.getpixel((x_index, y_index))[:3]) / 3
            for y_index in range(image.size[1])
        ]
        for x_index in range(image.size[0])
    ]

    # Matriz com valores de brilho (não-parametrizados, estão no range -510 a 510) para cada pixel após a aplicação do
    #   kernel.
    out_brightness = kernel.apply(
        function=lambda coord: brightness[coord[0]][coord[1]],
        limits=image.size,
        weight=1,
        default=0
    )

    # Criar a imagem nova e colocar os valores do brilho em cada pixel, após serem parametrizados. A parametrização é
    #   um simples módulo do brilho dividido por 2, e todos os pixels estão em escala de cinza.
    out_image = Image.new(mode="RGBA", size=image.size)

    for pixel_iy in range(out_image.size[1]):
        for pixel_ix in range(out_image.size[0]):
            out_image.putpixel(
                xy=(pixel_ix, pixel_iy),
                value=(
                    int(abs(out_brightness[pixel_ix][pixel_iy]) / 2),
                    int(abs(out_brightness[pixel_ix][pixel_iy]) / 2),
                    int(abs(out_brightness[pixel_ix][pixel_iy]) / 2),
                    255 if image.mode == "RGB" else image.getpixel((pixel_ix, pixel_iy))[3]
                )
            )

    return out_image


def box_blur(image: Image.Image) -> Image.Image:
    """Cria uma versão borrada da imagem passada.

    O algoritmo de borragem é o "box blur", e utiliza o kernel de convolução passado no vídeo relacionado à tarefa.

    Parâmetros
    ----------
    image : Image.Image
        A imagem a partir da qual a nova imagem borrada será gerada.

        A imagem deve ser uma imagem da biblioteca PIL (ou Pillow), e deve ter formato RGB ou RGBA [err #1].

    Retorno
    -------
    Image.Image
        A imagem borrada, gerada a partir da imagem passada. É uma imagem da biblioteca PIL (ou Pillow) em formato RGBA.

    Erros
    -----
    ValueError
    [1] Caso a imagem passada esteja em um formato que não seja RGB ou RGBA.
    """
    # Verificar se o formato da imagem está correto.
    if image.mode != "RGB" and image.mode != "RGBA":
        raise ValueError("[1] O formato da imagem deve ser RGB ou RGBA.")

    # Construir o kernel de convolução.
    kernel = ConvolutionKernel(
        matrix=[[0, 1, 0], [1, 0, 1], [0, 1, 0]],
        anchor=(1, 1)
    )

    # Aplicar o kernel de convolução correspondente à borragem em cada layer da imagem, exceto a transparência, que será
    #   conservada da imagem original.
    out_r = kernel.apply(function=lambda coord: image.getpixel(coord)[0], limits=image.size, weight=4, default=0)
    out_g = kernel.apply(function=lambda coord: image.getpixel(coord)[1], limits=image.size, weight=4, default=0)
    out_b = kernel.apply(function=lambda coord: image.getpixel(coord)[2], limits=image.size, weight=4, default=0)

    # Criar a imagem nova e colocar os novos valores de cada layer em cada pixel.
    out_image = Image.new(mode="RGBA", size=image.size)

    for pixel_iy in range(out_image.size[1]):
        for pixel_ix in range(out_image.size[0]):
            out_image.putpixel(
                xy=(pixel_ix, pixel_iy),
                value=(
                    int(out_r[pixel_ix][pixel_iy]),
                    int(out_g[pixel_ix][pixel_iy]),
                    int(out_b[pixel_ix][pixel_iy]),
                    255 if image.mode == "RGB" else image.getpixel((pixel_ix, pixel_iy))[3]
                )
            )

    return out_image


def gaussian_blur(image: Image.Image) -> Image.Image:
    """Cria uma versão borrada da imagem passada.

    O algoritmo de borragem é o "gaussian blur", e utiliza o kernel de convolução passado no vídeo relacionado à tarefa.

    Parâmetros
    ----------
    image : Image.Image
        A imagem a partir da qual a nova imagem borrada será gerada.

        A imagem deve ser uma imagem da biblioteca PIL (ou Pillow), e deve ter formato RGB ou RGBA [err #1].

    Retorno
    -------
    Image.Image
        A imagem borrada, gerada a partir da imagem passada. É uma imagem da biblioteca PIL (ou Pillow) em formato RGBA.

    Erros
    -----
    ValueError
    [1] Caso a imagem passada esteja em um formato que não seja RGB ou RGBA.
    """
    # Verificar se o formato da imagem está correto.
    if image.mode != "RGB" and image.mode != "RGBA":
        raise ValueError("[1] O formato da imagem deve ser RGB ou RGBA.")

    # Construir o kernel de convolução.
    kernel = ConvolutionKernel(
        matrix=[[1, 2, 1], [2, 4, 2], [1, 2, 1]],
        anchor=(1, 1)
    )

    # Aplicar o kernel de convolução correspondente à borragem em cada layer da imagem, exceto a transparência, que será
    #   conservada da imagem original.
    out_r = kernel.apply(function=lambda coord: image.getpixel(coord)[0], limits=image.size, weight=16, default=0)
    out_g = kernel.apply(function=lambda coord: image.getpixel(coord)[1], limits=image.size, weight=16, default=0)
    out_b = kernel.apply(function=lambda coord: image.getpixel(coord)[2], limits=image.size, weight=16, default=0)

    # Criar a imagem nova e colocar os novos valores de cada layer em cada pixel.
    out_image = Image.new(mode="RGBA", size=image.size)

    for pixel_iy in range(out_image.size[1]):
        for pixel_ix in range(out_image.size[0]):
            out_image.putpixel(
                xy=(pixel_ix, pixel_iy),
                value=(
                    int(out_r[pixel_ix][pixel_iy]),
                    int(out_g[pixel_ix][pixel_iy]),
                    int(out_b[pixel_ix][pixel_iy]),
                    255 if image.mode == "RGB" else image.getpixel((pixel_ix, pixel_iy))[3]
                )
            )

    return out_image


def sharpen(image: Image.Image) -> Image.Image:
    """Cria uma versão "afiada" da imagem passada.

    O algoritmo utilizado é o "sharpen", e utiliza o kernel de convolução passado no vídeo relacionado à tarefa.

    Parâmetros
    ----------
    image : Image.Image
        A imagem a partir da qual a nova imagem será gerada.

        A imagem deve ser uma imagem da biblioteca PIL (ou Pillow), e deve ter formato RGB ou RGBA [err #1].

    Retorno
    -------
    Image.Image
        A imagem "afiada", gerada a partir da imagem passada. É uma imagem da biblioteca PIL (ou Pillow) em formato
          RGBA.

    Erros
    -----
    ValueError
    [1] Caso a imagem passada esteja em um formato que não seja RGB ou RGBA.
    """
    # Verificar se o formato da imagem está correto.
    if image.mode != "RGB" and image.mode != "RGBA":
        raise ValueError("[1] O formato da imagem deve ser RGB ou RGBA.")

    # Construir o kernel de convolução.
    kernel = ConvolutionKernel(
        matrix=[[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
        anchor=(1, 1)
    )

    # Aplicar o kernel de convolução em cada layer da imagem, exceto a transparência, que será conservada da imagem
    #   original.
    out_r = kernel.apply(function=lambda coord: image.getpixel(coord)[0], limits=image.size, weight=1, default=0)
    out_g = kernel.apply(function=lambda coord: image.getpixel(coord)[1], limits=image.size, weight=1, default=0)
    out_b = kernel.apply(function=lambda coord: image.getpixel(coord)[2], limits=image.size, weight=1, default=0)

    # Criar a imagem nova e colocar os novos valores de cada layer em cada pixel.
    out_image = Image.new(mode="RGBA", size=image.size)

    for pixel_iy in range(out_image.size[1]):
        for pixel_ix in range(out_image.size[0]):
            out_image.putpixel(
                xy=(pixel_ix, pixel_iy),
                value=(
                    int(out_r[pixel_ix][pixel_iy]),
                    int(out_g[pixel_ix][pixel_iy]),
                    int(out_b[pixel_ix][pixel_iy]),
                    255 if image.mode == "RGB" else image.getpixel((pixel_ix, pixel_iy))[3]
                )
            )

    return out_image


def embossing(image: Image.Image) -> Image.Image:
    """Cria uma versão metálica da imagem passada.

    O algoritmo utilizado é o "embossing", e utiliza o kernel de convolução passado no vídeo relacionado à tarefa.

    Parâmetros
    ----------
    image : Image.Image
        A imagem a partir da qual a nova imagem será gerada.

        A imagem deve ser uma imagem da biblioteca PIL (ou Pillow), e deve ter formato RGB ou RGBA [err #1].

    Retorno
    -------
    Image.Image
        A imagem "metálica", gerada a partir da imagem passada. É uma imagem da biblioteca PIL (ou Pillow) em formato
          RGBA.

    Erros
    -----
    ValueError
    [1] Caso a imagem passada esteja em um formato que não seja RGB ou RGBA.
    """
    # Verificar se o formato da imagem está correto.
    if image.mode != "RGB" and image.mode != "RGBA":
        raise ValueError("[1] O formato da imagem deve ser RGB ou RGBA.")

    # Construir o kernel de convolução.
    kernel = ConvolutionKernel(
        matrix=[[0, 1, 1], [-1, 0, 1], [-1, -1, 0]],
        anchor=(1, 1)
    )

    # Aplicar o kernel de convolução em cada layer da imagem, exceto a transparência, que será conservada da imagem
    #   original.
    out_r = kernel.apply(function=lambda coord: image.getpixel(coord)[0], limits=image.size, weight=1, default=0)
    out_g = kernel.apply(function=lambda coord: image.getpixel(coord)[1], limits=image.size, weight=1, default=0)
    out_b = kernel.apply(function=lambda coord: image.getpixel(coord)[2], limits=image.size, weight=1, default=0)

    # Criar a imagem nova e colocar os novos valores de cada layer em cada pixel, após serem parametrizados. A
    #   aplicação do kernel terá resultados entre -765 e +765, que serão parametrizados linearmente para o intervalo
    #   0 a +255.
    out_image = Image.new(mode="RGBA", size=image.size)

    for pixel_iy in range(out_image.size[1]):
        for pixel_ix in range(out_image.size[0]):
            out_image.putpixel(
                xy=(pixel_ix, pixel_iy),
                value=(
                    int((out_r[pixel_ix][pixel_iy] + 765) / 6),
                    int((out_g[pixel_ix][pixel_iy] + 765) / 6),
                    int((out_b[pixel_ix][pixel_iy] + 765) / 6),
                    255 if image.mode == "RGB" else image.getpixel((pixel_ix, pixel_iy))[3]
                )
            )

    return out_image
