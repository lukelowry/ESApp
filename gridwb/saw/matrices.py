import os
import re
import tempfile
from pathlib import Path
from typing import Union

import numpy as np
from scipy.sparse import csr_matrix


class MatrixMixin:
    
    def get_ybus(self, full: bool = False, file: Union[str, None] = None) -> Union[np.ndarray, csr_matrix]:
        """Obtain the YBus matrix from PowerWorld.

        This method calls the ``SaveYbusInMatlabFormat`` script command
        to write the YBus to a temporary file, then parses that file
        to construct the matrix.

        :param full: If True, returns a dense NumPy array. If False
            (default), returns a SciPy CSR sparse matrix.
        :param file: Optional path to a pre-existing ``.mat`` file. If
            provided, the file will be parsed directly instead of
            calling SimAuto to generate a new one.
        :return: The YBus matrix as either a dense NumPy array or a
            SciPy CSR sparse matrix.
        """
        if file:
            _tempfile_path = file
        else:
            _tempfile = tempfile.NamedTemporaryFile(mode="w", suffix=".mat", delete=False)
            _tempfile_path = Path(_tempfile.name).as_posix()
            _tempfile.close()
            cmd = f'SaveYbusInMatlabFormat("{_tempfile_path}", NO)'
            self.RunScriptCommand(cmd)
        with open(_tempfile_path, "r") as f:
            f.readline()
            mat_str = f.read()
        mat_str = re.sub(r"\s", "", mat_str)
        lines = re.split(";", mat_str)
        ie = r"[0-9]+"
        fe = r"-*[0-9]+\.[0-9]+"
        dr = re.compile(r"(?:Ybus)=(?:sparse\()({ie})".format(ie=ie))
        exp = re.compile(r"(?:Ybus\()({ie}),({ie})(?:\)=)({fe})(?:\+j\*)(?:\()({fe})".format(ie=ie, fe=fe))
        dim = dr.match(lines[0])[1]
        n = int(dim)
        row, col, data = [], [], []
        for line in lines[1:]:
            match = exp.match(line)
            if match is None:
                continue
            idx1, idx2, real, imag = match.groups()
            admittance = float(real) + 1j * float(imag)
            row.append(int(idx1))
            col.append(int(idx2))
            data.append(admittance)

        sparse_matrix = csr_matrix(
            (data, (np.asarray(row) - 1, np.asarray(col) - 1)),
            shape=(n, n),
            dtype=complex,
        )
        return sparse_matrix.toarray() if full else sparse_matrix

    def get_branch_admittance(self):
        """Calculate the branch admittance matrices, Yf and Yt.

        These matrices describe the relationship between branch currents
        and bus voltages. This method calculates them based on branch
        parameters retrieved from the current case.

        :return: A tuple containing two SciPy CSR sparse matrices: (Yf, Yt).
        """
        key = self.get_key_field_list("bus")
        df = self.GetParametersMultipleElement("bus", key)

        branch = self.GetParametersMultipleElement(
            "branch",
            self.get_key_field_list("branch") + ["LineR", "LineX", "LineC", "LineTap", "LinePhase"],
        )
        branch["LineR"] = branch["LineR"].astype(float)
        branch["LineX"] = branch["LineX"].astype(float)
        branch["LineC"] = branch["LineC"].astype(float)
        branch["LineTap"] = branch["LineTap"].astype(float)
        branch["LinePhase"] = branch["LinePhase"].astype(float)

        nb = df.shape[0]
        nl = branch.shape[0]

        Ys = 1 / (branch["LineR"].to_numpy() + 1j * branch["LineX"].to_numpy())
        Bc = branch["LineC"].to_numpy()
        tap = branch["LineTap"].to_numpy() * np.exp(1j * np.pi / 180 * branch["LinePhase"].to_numpy())
        Ytt = Ys + 1j * Bc / 2
        Yff = Ytt / (tap * np.conj(tap))
        Yft = -Ys / np.conj(tap)
        Ytf = -Ys / tap

        def loop_translate(a, d):
            n = np.ndarray(a.shape, dtype=int)
            for k, v in d.items():
                n[a == k] = v
            return n

        d = {value: index for index, value in df["BusNum"].items()}
        f = loop_translate(branch["BusNum"].to_numpy(dtype=int).reshape(-1), d)
        t = loop_translate(branch["BusNum:1"].to_numpy(dtype=int).reshape(-1), d)

        i = np.r_[range(nl), range(nl)]
        Yf = csr_matrix((np.hstack([Yff.reshape(-1), Yft.reshape(-1)]), (i, np.hstack([f, t]))), (nl, nb))
        Yt = csr_matrix((np.hstack([Ytf.reshape(-1), Ytt.reshape(-1)]), (i, np.hstack([f, t]))), (nl, nb))
        return Yf, Yt

    def get_shunt_admittance(self):
        """Calculate the shunt admittance vector, Ysh.

        This vector represents the admittance to ground for each bus
        from shunt elements.

        :return: A NumPy array representing the shunt admittance for each bus.
        """
        base = self.GetParametersMultipleElement("Sim_Solution_Options", ["SBase"]).to_numpy(float).ravel()
        key = self.get_key_field_list("bus")
        df = self.GetParametersMultipleElement("bus", key + ["BusSS", "BusSSMW"])
        df["BusSS"] = df["BusSS"].astype(float)
        df["BusSSMW"] = df["BusSSMW"].astype(float)
        df.fillna(0, inplace=True)
        return (df["BusSSMW"].to_numpy() + 1j * df["BusSS"].to_numpy()) / base

    def get_gmatrix(self, full: bool = False) -> Union[np.ndarray, csr_matrix]:
        """Get the GIC conductance matrix (G).

        This method calls the ``GICSaveGMatrix`` script command to
        write the G-matrix to a temporary file, then parses that file
        to construct the matrix.

        :param full: If True, returns a dense NumPy array. If False
            (default), returns a SciPy CSR sparse matrix.
        :return: The G-matrix as either a dense NumPy array or a
            SciPy CSR sparse matrix.
        """
        g_matrix_path, id_file_path = self._make_temp_matrix_files()
        try:
            cmd = f'GICSaveGMatrix("{g_matrix_path}","{id_file_path}");'
            self.RunScriptCommand(cmd)
            self.RunScriptCommand(cmd)
            with open(g_matrix_path, "r") as f:
                mat_str = f.read()
            sparse_matrix = self._parse_real_matrix(mat_str, "GMatrix")
            return sparse_matrix.toarray() if full else sparse_matrix
        finally:
            os.unlink(g_matrix_path)
            os.unlink(id_file_path)

    def get_jacobian(self, full: bool = False) -> Union[np.ndarray, csr_matrix]:
        """Get the power flow Jacobian matrix.

        This method calls the ``SaveJacobian`` script command to write
        the Jacobian to a temporary file, then parses that file to
        construct the matrix.

        :param full: If True, returns a dense NumPy array. If False
            (default), returns a SciPy CSR sparse matrix.
        :return: The Jacobian matrix as either a dense NumPy array or a
            SciPy CSR sparse matrix.
        """
        jac_file_path, id_file_path = self._make_temp_matrix_files()
        try:
            cmd = f'SaveJacobian("{jac_file_path}","{id_file_path}",M,R);'
            self.RunScriptCommand(cmd)
            with open(jac_file_path, "r") as f:
                mat_str = f.read()
            sparse_matrix = self._parse_real_matrix(mat_str, "Jac")
            return sparse_matrix.toarray() if full else sparse_matrix
        finally:
            os.unlink(jac_file_path)
            os.unlink(id_file_path)

    def get_incidence_matrix(self):
        """Calculate the bus-branch incidence matrix.

        The incidence matrix (A) describes the topology of the network,
        with entries A[i, j] = 1 if branch i starts at bus j, -1 if
        branch i ends at bus j, and 0 otherwise.

        :return: A NumPy array representing the incidence matrix.
        """
        branch = self.ListOfDevices("branch")
        bus = self.ListOfDevices("bus")
        incidence = np.zeros([branch.shape[0], bus.shape[0]], dtype=int)
        for i, row in branch.iterrows():
            incidence[i, row["BusNum"] - 1] = 1
            incidence[i, row["BusNum:1"] - 1] = -1
        return incidence

    def _make_temp_matrix_files(self):
        mat_file = tempfile.NamedTemporaryFile(mode="w", suffix=".m", delete=False)
        mat_file_path = Path(mat_file.name).as_posix()
        mat_file.close()
        id_file = tempfile.NamedTemporaryFile(mode="w", delete=False)
        id_file_path = Path(id_file.name).as_posix()
        id_file.close()
        return mat_file_path, id_file_path

    def _parse_real_matrix(self, mat_str, matrix_name="Jac"):
        """Helper to parse a real-valued sparse matrix from '.m' PowerWorld output."""
        mat_str = re.sub(r"\s", "", mat_str)
        lines = re.split(";", mat_str)
        ie = r"[0-9]+"
        fe = r"-*[0-9]+\.[0-9]+"
        dr = re.compile(r"(?:{matrix_name})=(?:sparse\()({ie})".format(ie=ie, matrix_name=matrix_name))
        exp = re.compile(r"(?:{matrix_name}\()({ie}),({ie})(?:\)=)({fe})".format(ie=ie, fe=fe, matrix_name=matrix_name))
        dim = dr.match(lines[0])[1]
        n = int(dim)
        row, col, data = [], [], []
        for line in lines[1:]:
            match = exp.match(line)
            if match is None:
                continue
            idx1, idx2, real = match.groups()
            row.append(int(idx1))
            col.append(int(idx2))
            data.append(float(real))
        return csr_matrix((data, (np.asarray(row) - 1, np.asarray(col) - 1)), shape=(n, n))