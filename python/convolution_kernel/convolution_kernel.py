"""Define a classe 'ConvolutionKernel'."""
# Importações # ------------------------------------------------------------------------------------------------------ #
# Módulos
from typing import Callable, List, Optional, Sequence, Tuple


# Classes # ---------------------------------------------------------------------------------------------------------- #
class ConvolutionKernel:
    # TODO - ConvolutionKernel: a docstring desta classe.
    # Atributos # ---------------------------------------------------------------------------------------------------- #
    __slots__ = ["_matrix", "_anchor"]
    _matrix: List[List[float]]
    _anchor: Tuple[int, int]

    # Construtores # ------------------------------------------------------------------------------------------------- #
    def __init__(self, matrix: Sequence[Sequence[float]], anchor: Optional[Tuple[int, int]] = None) -> None:
        """
        Parâmetros
        ----------
        matrix : Sequence[Sequence[float]]
            Uma variável que pode ser interpretada como uma matriz de números reais, contendo os valores para o kernel
              de convolução.

            Não é possível mudar as dimensões da matriz após a instanciação da classe.

            A âncora da matriz para a aplicação do kernel de convolução é definida no parâmetro 'anchor'.

            As sequências internas deste parâmetro devem todas ter o mesmo tamanho, para que este parâmetro represente
              uma matriz válida [err #1].
        anchor : Optional[Tuple[int, int]] = None
            A posição da âncora na matriz, representada por coordenadas (x, y) onde x é o índice da coluna e y é o
              índice da linha, para a aplicação do kernel de convolução.

            Caso seu valor seja 'None', o método tentará determinar uma âncora centralizada (e.g. uma matriz 3x3 terá
              âncora nos índices (1, 1)), o que só terá sucesso caso a matriz tenha tamanhos ímpares. Caso uma das
              dimensões da matriz tenha tamanho par, a âncora não poderá ser determinada [err #2].

            Caso uma posição seja atribuída para a âncora, ela deve representar uma posição válida da matriz. Note que
              para uma matriz MxN, suas primeiras linhas e colunas têm índice 0, que cresce por 1 para cada próxima
              linha/coluna, logo as últimas linha/coluna têm índices M - 1 e N - 1, respectivamente [err #3].

        Erros
        -----
        ValueError
        [1] Se o parâmetro 'matrix' tiver sequências internas com tamanhos diferentes.

        [2] Se o parâmetro 'anchor' for definido como 'None' e a matriz representada pelo parâmetro 'matrix' tenha pelo
              menos uma dimensão par.

        IndexError
        [3] Se o parâmetro 'anchor' for definido como uma tupla de coordenadas e tais coordenadas estiverem fora das
              dimensões da matriz.
        """
        # Verificar se o parâmetro 'matrix' representa uma matriz válida.
        if not all(len(inner_sequence) == len(matrix[0]) for inner_sequence in matrix[1:]):
            raise ValueError("[1] Parâmetro 'matrix' não representa uma matriz válida.")
        if anchor is None:
            # Verificar se é possível calcular uma âncora centralizada.
            if len(matrix) % 2 == 0 or len(matrix[0]) % 2 == 0:
                raise ValueError("[2] Não é possível determinar uma âncora para a matriz passada.")
            else:
                # Calcular a âncora.
                anchor = ((len(matrix) - 1) // 2, (len(matrix[0]) - 1) // 2)
        else:
            # Verificar se a âncora passada está dentro dos limites da matriz.
            if not 0 <= anchor[0] < len(matrix) or not 0 <= anchor[1] < len(matrix[0]):
                raise IndexError("[3] Âncora representa posição fora da matriz.")

        # Inicializar o objeto.
        # A matriz deve ser copiada para não ser modificada externamente.
        self._matrix = [[matrix[col][row] for row in range(len(matrix[0]))] for col in range(len(matrix))]
        self._anchor = anchor

    # Propriedades # ------------------------------------------------------------------------------------------------- #
    @property
    def anchor(self) -> Tuple[int, int]:
        """A posição da matriz que atua como âncora na aplicação do kernel de convolução.

        Retorno
        -------
        Tuple[int, int]
            Uma tupla de dois inteiros, representando as coordenadas (x, y) da âncora na matriz.
        """
        return self._anchor

    @anchor.setter
    def anchor(self, value: Tuple[int, int]) -> None:
        """
        Parâmetros
        ----------
        value : Tuple[int, int]
            As coordenadas (x, y) da nova posição na matriz que será utilizada como âncora na aplicação do kernel de
              convolução.

            Deve representar uma posição válida na matriz. Isto é, para uma matriz MxN, as coordenadas da âncora devem
              satisfazer 0 <= x < M e 0 <= y < N [err #1].

        Erros
        -----
        IndexError
        [1] Se as coordenadas passadas estiverem fora das dimensões da matriz.
        """
        # Verificar se a âncora passada está fora dos limites da matriz.
        if not 0 <= value[0] < self.width or not 0 <= value[1] < self.height:
            raise IndexError("[1] Âncora representa posição fora da matriz.")

        self._anchor = value

    @property
    def width(self) -> int:
        """A largura da matriz (número de colunas) correspondente ao kernel de convolução.

        Retorno
        -------
        int
            A largura da matriz correspondente ao kernel de convolução.
        """
        return len(self._matrix)

    @property
    def height(self) -> int:
        """A altura da matriz (número de linhas) correspondente ao kernel de convolução.

        Retorno
        -------
        int
            A altura da matriz correspondente ao kernel de convolução.
        """
        return len(self._matrix[0])

    @property
    def sum(self) -> int:
        """A soma de todos os elementos na matriz correspondente ao kernel de convolução.

        Retorno
        -------
        int
            A soma de todos os elementos na matriz correspondente ao kernel de convolução.
        """
        return sum([sum([num for num in col]) for col in self._matrix])

    # Operadores # --------------------------------------------------------------------------------------------------- #
    def __getitem__(self, index: Tuple[int, int]) -> float:
        """Acessa uma posição da matriz correspondente ao kernel de convolução.

        Parâmetros
        ----------
        index : Tuple[int, int]
            A posição na matriz do elemento que se deseja acessar, no formato (x, y), onde x é o índice da coluna e y é
              o índice da linha.

            A indexação horizontal se inicia na esquerda, e a vertical no topo, ambos com índice 0 que cresce em 1 para
              cada elemento. Logo, em uma matriz N x M, os índices horizontais vão de 0 a N - 1, e os verticais de 0 a
              M - 1, e os índices passados como argumento devem respeitar esses limites [err #1].

        Retorno
        -------
        float
            O valor da matriz, correspondente ao kernel de convolução, na posição passada.

        Erros
        -----
        IndexError
        [1] Caso o parâmetro 'index' represente uma posição fora das dimensões da matriz.
        """
        # Verificar se a posição especificada está dentro dos limites permitidos.
        if not 0 <= index[0] < self.width or not 0 <= index[1] < self.height:
            raise IndexError("[1] Parâmetro 'index' representa posição fora da matriz.")

        return self._matrix[index[0]][index[1]]

    def __setitem__(self, index: Tuple[int, int], value: float) -> None:
        # TODO - ConvolutionKernel.__setitem__: a docstring deste operador.
        # Verificar se a posição especificada está dentro dos limites permitidos.
        if not 0 <= index[0] < self.width or not 0 <= index[1] < self.height:
            raise IndexError("[1]")

        self._matrix[index[0]][index[1]] = value

    # Métodos # ------------------------------------------------------------------------------------------------------ #
    def apply(self, function: Callable[[Tuple[int, int]], float], limits: Tuple[int, int], weight: int = 1,
              default: int = 0) -> List[List[float]]:
        """Aplica o kernel de convolução em uma série de dados que representam uma matriz.

        Parâmetros
        ----------
        function : Callable[[Tuple[int, int]], float]
            Um objeto chamável responsável pelo acesso aos valores da matriz que será processada.

            Deve receber uma tupla de dois inteiros, representando as coordenadas (x, y) do elemento da matriz que será
              acessado, e retornar o valor da matriz nessas coordenadas.

            Os limites de parâmetro para as coordenadas que serão passadas para essa "função" devem ser definidos no
              parâmetro 'limits', e espera-se que essa "função" tenha valores definidos nos intervalos
              0 <= x < limits[0] e 0 <= y < limits[1], para coordenadas (x, y). Caso isto não aconteça, resultará em
              algum erro proveniente da "função" passada.
        limits : Tuple[int, int]
            Os limites de iteração para os dados que representam a matriz que será processada.

            Espera-se que a "função" representada pelo parâmetro 'function' esteja bem definida nos intervalos
              0 <= x < limits[0] e 0 <= y < limits[1], para coordenadas (x, y). Caso isto não aconteça, provavelmente
              resultará em algum erro proveniente da "função" passada.
        weight : int
            O peso pelo qual o kernel de convolução deverá dividir a soma ponderada dos valores relacionados para cada
              posição da matriz que será processada.

            e.g.: um kernel de convolução representado pela matriz de uma única linha [1, 0, 1] irá calcular a soma de
              valores à esquerda e à direita de cada valor na matriz de dados a ser processada. Caso o peso seja igual a
              2, isto mudaria o propósito do kernel para calcular a média dos valores, ao invés da soma.

            Considere utilizar 'self.sum' para kernels que calculam médias ponderadas.
        default : int
            O valor padrão para ser utilizado quando um dado valor na matriz de dados não possuir vizinhos o suficiente.
              O valor do vizinho será substituído por este valor.

            e.g.: um kernel de convolução representado por uma matriz 3x3 com âncora no centro, quando aplicado ao valor
              na posição (0, 0) da matriz de dados, não terá como obter os valores nas posições (-1, -1), (-1, 0),
              (-1, 1), (0, -1) e (1, -1). Estes valores serão substituídos por este valor padrão.

        Retorna
        -------
        List[List[float]]
            Uma matriz com valores correspondentes à aplicação do kernel na matriz de dados, para cada posição.
        """
        # Matriz que será retornada.
        output: List[List[float]] = [[0 for _ in range(limits[1])] for _ in range(limits[0])]

        # Itera-se sobre cada valor da estrutura original, linha por linha, e calcula-se o novo valor.
        for index_y in range(limits[1]):
            for index_x in range(limits[0]):
                new_total = 0

                # Aplicar o kernel de convolução.
                for kernel_iy in range(self.height):
                    for kernel_ix in range(self.width):
                        target_ix = index_x + kernel_ix - (self.width - 1) // 2
                        target_iy = index_y + kernel_iy - (self.height - 1) // 2

                        # Se a posição buscada estiver fora dos limites da imagem, utilizar o valor padrão.
                        if 0 <= target_ix < limits[0] and 0 <= target_iy < limits[1]:
                            # Aplicar o peso do kernel naquela posição e somar o resultado ao total.
                            new_total += function((target_ix, target_iy)) * self[kernel_ix, kernel_iy]
                        else:
                            new_total += default * self[kernel_ix, kernel_iy]

                # Dividir o total pelo peso e colocá-lo na matriz que será retornada.
                output[index_x][index_y] = new_total / weight

        return output
