{
  description = "Development environment for rescue-python";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    { nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python314;
      in
      {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            git
            go-task
            jdk21
            python
            uv
          ];

          env = {
            # Use Nix's Python instead of letting uv download another interpreter.
            UV_PYTHON = "${python}/bin/python";
            UV_PYTHON_DOWNLOADS = "never";
          };

          shellHook = ''
            echo "rescue-python development shell"
            echo "Python: $(python --version)"
            echo "uv: $(uv --version)"
            echo "Task: $(task --version)"
          '';
        };
      }
    );
}
