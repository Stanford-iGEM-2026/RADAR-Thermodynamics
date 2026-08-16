import streamlit as st
import subprocess
import zipfile
import tempfile
from pathlib import Path

st.title("Gene → FASTA")
st.write("Enter a human gene symbol and get its transcript FASTA from NCBI.")

gene = st.text_input("Gene symbol", value="ADAM12").strip().upper()

if st.button("Get FASTA"):
    if not gene:
        st.error("Enter a gene symbol.")
    else:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)

            zip_path = tmp / f"{gene}.zip"

            command = [
                "datasets",
                "download",
                "gene",
                "symbol",
                gene,
                "--taxon",
                "human",
                "--include",
                "rna",
                "--filename",
                str(zip_path),
            ]

            try:
                subprocess.run(command, check=True, capture_output=True, text=True)

                with zipfile.ZipFile(zip_path, "r") as z:
                    z.extractall(tmp / "unzipped")

                fasta_files = list((tmp / "unzipped").rglob("rna.fna"))

                if not fasta_files:
                    st.error("NCBI returned no transcript FASTA for this gene.")
                else:
                    fasta_text = fasta_files[0].read_text()

                    st.success(f"Found transcript FASTA for {gene}")

                    st.text_area(
                        "FASTA",
                        fasta_text,
                        height=400,
                    )

                    st.download_button(
                        label="Download FASTA",
                        data=fasta_text,
                        file_name=f"{gene}_transcripts.fasta",
                        mime="text/plain",
                    )

            except FileNotFoundError:
                st.error(
                    "NCBI Datasets CLI is not installed. "
                    "Install it first, then restart the app."
                )

            except subprocess.CalledProcessError as e:
                st.error("NCBI could not retrieve that gene.")
                st.code(e.stderr)
