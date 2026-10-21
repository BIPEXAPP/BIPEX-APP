# GENJUTSU AI (primeira versão)

Plataforma privada de geração de imagens. Vídeo ainda NÃO está implementado (precisa de GPU).

## Estrutura
- `index.html`: frontend estático (GitHub Pages).
- `server/app.py`: API com fila, estados reais e histórico (SQLite).
- `tests/`: testes da API (`pytest tests`). Usam um adaptador falso só para testar a fila.

## Rodar o servidor
1. `cd server && pip install -r requirements.txt -r requirements-ai.txt` (instale o torch conforme o seu hardware).
2. Defina um token: `export GENJUTSU_TOKEN="uma-frase-longa-e-secreta"`.
3. `uvicorn app:app --host 127.0.0.1 --port 8000`.
4. Para o site no GitHub Pages falar com ele, o servidor precisa de um endereço HTTPS. Não exponha a porta direto à internet.
5. No site: Configurações, informe a URL e o token.

## Limitações honestas
- Sem GPU, a imagem demora minutos. O modelo padrão é `stabilityai/sd-turbo`; a licença NÃO foi verificada. Confira antes de uso comercial. Troque com `GENJUTSU_IMAGE_MODEL`.
- Só tarefas na fila podem ser canceladas. Não há limite de requisições nem contas de usuário ainda (token único).
- A geração real de imagem NÃO foi testada em um computador com o modelo baixado.
