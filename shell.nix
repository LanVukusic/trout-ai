{ pkgs ? import <nixpkgs> {} }:
let
  my-python-packages = p: with p; [
    jupyter
    ipykernel
    ipython
    numpy
    torch
    opencv4
    matplotlib
    chess
    # other python packages
  ];
  my-python = pkgs.python3.withPackages my-python-packages;

  my-system-packages = p: with p; [
    # other system packages
    pkgs.wget
    pkgs.zstd

  ];

  # we need wget and zstd to download the model
  my-python-env = pkgs.buildEnv {
    name = "my-python-env";
    paths = [ my-python pkgs.wget pkgs.zstd ];
  };
in
pkgs.mkShell {
  name = "chess ai dev shell";
  buildInputs = [ my-python-env ];
}
