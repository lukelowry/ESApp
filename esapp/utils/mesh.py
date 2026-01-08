"""

Handles .ply mesh file reading and writing.

And conversions to useful formats.

"""

from dataclasses import dataclass
import numpy as np
import struct
from scipy.sparse import csc_matrix

def extract_unique_edges(faces):
    """
    Extracts a sorted list of unique edges from a list of faces.
    
    Parameters
    ----------
    faces : list of lists
        The mesh faces.
        
    Returns
    -------
    np.ndarray
        An (M, 2) array of unique edges where col 0 < col 1.
    """
    unique_edges = set()
    
    for face in faces:
        n = len(face)
        for i in range(n):
            u = face[i]
            v = face[(i + 1) % n]  # Connect to next vertex
            # Sort pair to ensure (u, v) is same as (v, u)
            edge = (u, v) if u < v else (v, u)
            unique_edges.add(edge)
            
    # Return as a sorted numpy array for consistent indexing
    return np.array(sorted(list(unique_edges)), dtype=int)


@dataclass
class Mesh:
    vertices: list[tuple[float, float, float]]
    faces: list[list[int]]

    @classmethod
    def from_ply(cls, filepath: str) -> "Mesh":
        """
        Reads a .ply file and constructs a Mesh object.
        
        Parameters
        ----------
        filepath : str
            Path to the .ply file.

        Returns
        -------
        Mesh
            The constructed Mesh object.
        """
        with open(filepath, 'rb') as f:
            # Parse Header
            header_ended = False
            fmt = "ascii"
            vertex_count = 0
            face_count = 0
            vertex_props = []
            current_element = None

            while not header_ended:
                line = f.readline().strip()
                if not line:
                    break
                line_str = line.decode('ascii', errors='ignore')

                if line_str == "end_header":
                    header_ended = True
                    break

                parts = line_str.split()
                if not parts:
                    continue

                if parts[0] == "format":
                    fmt = parts[1]
                elif parts[0] == "element":
                    current_element = parts[1]
                    if current_element == "vertex":
                        vertex_count = int(parts[2])
                    elif current_element == "face":
                        face_count = int(parts[2])
                elif parts[0] == "property":
                    if current_element == "vertex":
                        # parts[1] is type, parts[2] is name
                        vertex_props.append((parts[2], parts[1]))

            # Parse Body
            vertices = []
            faces = []

            if fmt == "ascii":
                lines = f.readlines()
                for i in range(vertex_count):
                    parts = lines[i].strip().split()
                    # Assume first 3 are x, y, z
                    v = (float(parts[0]), float(parts[1]), float(parts[2]))
                    vertices.append(v)
                
                for i in range(face_count):
                    parts = lines[vertex_count + i].strip().split()
                    vertex_indices = [int(x) for x in parts[1:]]
                    faces.append(vertex_indices)

            elif fmt == "binary_little_endian":
                np_type_map = {
                    'char': 'i1', 'uchar': 'u1', 'short': 'i2', 'ushort': 'u2',
                    'int': 'i4', 'uint': 'u4', 'float': 'f4', 'double': 'f8'
                }
                dtype_fields = [(name, np_type_map.get(type_str, 'f4')) for name, type_str in vertex_props]
                vertex_dtype = np.dtype(dtype_fields)
                
                vertex_data = f.read(vertex_count * vertex_dtype.itemsize)
                v_arr = np.frombuffer(vertex_data, dtype=vertex_dtype)
                
                # Extract x, y, z
                if 'x' in v_arr.dtype.names and 'y' in v_arr.dtype.names and 'z' in v_arr.dtype.names:
                    vertices = list(zip(v_arr['x'], v_arr['y'], v_arr['z']))
                else:
                    names = v_arr.dtype.names
                    vertices = list(zip(v_arr[names[0]], v_arr[names[1]], v_arr[names[2]]))

                for _ in range(face_count):
                    n = struct.unpack('<B', f.read(1))[0] # uchar count
                    faces.append(list(struct.unpack(f'<{n}i', f.read(n * 4)))) # int indices
            else:
                raise ValueError(f"Unsupported PLY format: {fmt}")

        return cls(vertices=vertices, faces=faces)

    def get_incidence_matrix(self) -> csc_matrix:
        """
        Constructs the sparse oriented incidence matrix B for the mesh.
        
        Returns
        -------
        scipy.sparse.csc_matrix
            Matrix B of size (\|V\| x \|E\|).
        """
        # Topological data
        vertices = self.vertices
        faces = self.faces

        # Extract unique edges
        edges = extract_unique_edges(faces)
        num_verts = len(vertices)
        num_edges = len(edges)

        # COO format data
        x = edges.ravel()
        y = np.repeat(np.arange(num_edges), 2)
        e = np.tile([1.0, -1.0], num_edges)
        
        # Construct Incidene matrix
        Bshp = (num_verts, num_edges)
        B = csc_matrix((e, (x, y)), shape=Bshp)
        
        return B

    def get_xyz(self) -> np.ndarray:
        """
        Returns the vertex coordinates as a numpy array.
        
        Returns
        -------
        np.ndarray
            An (N, 3) array of vertex coordinates.
        """
        return np.array(self.vertices)

    def to_laplacian(self) -> csc_matrix:
        """
        Constructs the graph Laplacian matrix L for the mesh.
        
        Returns
        -------
        scipy.sparse.csc_matrix
            The graph Laplacian matrix L.
        """
        B = self.get_incidence_matrix()
        L = B @ B.T
        return L
