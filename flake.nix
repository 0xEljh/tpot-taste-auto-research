{
  description = "tpot-taste: small-model post-training (SFT + RL) for tpot / tech-Twitter virality";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
    in {
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = with pkgs; [
          python312
          uv
          cacert

          git
          gcc
          gnumake
          pkg-config
          cmake
          ninja

          # Runtime libs that pip/uv-installed binary wheels dynamically link
          # against (torch, bitsandbytes, tokenizers, ...). NixOS has no global
          # /usr/lib, so these must be on LD_LIBRARY_PATH (see shellHook).
          stdenv.cc.cc.lib
          zlib
        ];

        shellHook = ''
          # --- WSL2 CUDA driver passthrough (libcuda.so lives outside the Nix store) ---
          if [ -d /usr/lib/wsl/lib ]; then
            export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$LD_LIBRARY_PATH"
          fi

          # Triton can fail to resolve libcuda via ldconfig on Nix systems.
          if [ -z "$TRITON_LIBCUDA_PATH" ]; then
            for d in /usr/lib/wsl/lib /usr/lib64-nvidia /run/opengl-driver/lib; do
              if [ -e "$d/libcuda.so.1" ]; then
                export TRITON_LIBCUDA_PATH="$d"
                break
              fi
            done
          fi

          [ -d /usr/lib64-nvidia ]    && export LD_LIBRARY_PATH="/usr/lib64-nvidia:$LD_LIBRARY_PATH"
          [ -d /usr/local/cuda/lib64 ] && export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

          # Expose Nix C/C++ runtime (libstdc++, libz, ...) to binary wheels.
          export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib pkgs.zlib ]}:$LD_LIBRARY_PATH"

          [ -z "$SSL_CERT_FILE" ] && export SSL_CERT_FILE="${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"

          # uv: use the Nix-provided Python; never silently download another.
          export UV_PYTHON_DOWNLOADS=never
          export UV_PYTHON="${pkgs.python312}/bin/python3.12"

          # Load project secrets / run config if present (WANDB_API_KEY, HF_HOME, ...).
          if [ -f .env ]; then
            set -a; . ./.env; set +a
          fi

          echo "tpot-taste dev shell ready."
          echo "  core deps:  uv sync"
          echo "  + modeling: uv sync --extra train"
        '';
      };
    };
}
