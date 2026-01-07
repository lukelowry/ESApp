"""

Handles .ply mesh file reading and writing.

And conversions to useful formats.

"""

from dataclasses import dataclass
import numpy as np
from scipy.sparse import csc_matrix

def extract_unique_edges(faces):
    """
    Extracts a sorted list of unique edges from a list of faces.
    
    Args:
        faces (list of lists): The mesh faces.
        
    Returns:
        np.ndarray: An (M, 2) array of unique edges where col 0 < col 1.
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
        
        Args:
            filepath (str): Path to the .ply file.
        Returns:
            Mesh: The constructed Mesh object.
        """
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Parse header
        vertex_count = 0
        face_count = 0
        header_ended = False
        line_idx = 0
        
        while not header_ended:
            line = lines[line_idx].strip()
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
            elif line.startswith("element face"):
                face_count = int(line.split()[-1])
            elif line == "end_header":
                header_ended = True
            line_idx += 1
        
        # Parse vertices
        vertices = []
        for i in range(vertex_count):
            parts = lines[line_idx + i].strip().split()
            vertex = (float(parts[0]), float(parts[1]), float(parts[2]))
            vertices.append(vertex)
        
        line_idx += vertex_count
        
        # Parse faces
        faces = []
        for i in range(face_count):
            parts = lines[line_idx + i].strip().split()
            vertex_indices = list(map(int, parts[1:]))  # Skip the first count element
            faces.append(vertex_indices)
        
        return cls(vertices=vertices, faces=faces)

    def get_incidence_matrix(self) -> csc_matrix:
        """
        Constructs the sparse oriented incidence matrix B for the mesh.
        
        Returns:
            scipy.sparse.csc_matrix: Matrix B of size (|V| x |E|).
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
        
        Returns:
            np.ndarray: An (N, 3) array of vertex coordinates.
        """
        return np.array(self.vertices)

    def to_laplacian(self) -> csc_matrix:
        """
        Constructs the graph Laplacian matrix L for the mesh.
        
        Returns:
            scipy.sparse.csc_matrix: The graph Laplacian matrix L.
        """
        B = self.get_incidence_matrix()
        L = B @ B.T
        return L
