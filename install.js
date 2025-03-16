module.exports = {
  run: [
    // Delete this step if your project does not use torch
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",                // Edit this to customize the venv folder path
          // xformers: true   // uncomment this line if your project requires xformers
        }
      }
    },
    // Edit this step with your custom install commands
    {
      method: "shell.run",
      params: {
        venv: "env",                // Edit this to customize the venv folder path
        message: [
          "uv pip install gradio devicetorch",
          "uv pip install -r requirements.txt"
        ],
      }
    },
    {
      method: "fs.link",
      params: {
        venv: "env"
      }
    },

    // espeak-ng installer script lifted from AllTalk Launcher from 6Morpheus6
    // https://github.com/pinokiofactory/AllTalk-TTS/blob/main/install.js
    {
      when: "{{which('brew')}}",
      method: "shell.run",
      params: {
        message: "brew install espeak-ng"
      },
      next: 'end'
    },
    {
      when: "{{which('apt')}}",
      method: "shell.run",
      params: {
        sudo: true,
        message: "apt install libaio-dev espeak-ng"
      },
      next: 'end'
    },
    {
      when: "{{which('yum')}}",
      method: "shell.run",
      params: {
        sudo: true,
        message: "yum install libaio-devel espeak-ng"
      },
      next: 'end'
    },
    {
      when: "{{which('winget')}}",
      method: "shell.run",
      params: {
        sudo: true,
        message: "winget install --id=eSpeak-NG.eSpeak-NG -e --silent --accept-source-agreements --accept-package-agreements"
      }
    },
    {
      id: 'end',
      method: 'input',
      params: {
        title: "Install Complete!!",
        description: "Install Complete."
      }
    },
  ]
}

