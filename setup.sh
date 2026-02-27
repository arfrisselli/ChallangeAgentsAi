#!/bin/bash
# Setup script: Cria .env a partir do template se não existir

set -e

echo "🚀 ChallangeAgentsAi - Setup"
echo ""

# 1. Verificar se .env existe
if [ -f .env ]; then
    echo "✅ Arquivo .env já existe"
else
    echo "📝 Criando .env a partir de .env.example..."
    cp .env.example .env
    echo "⚠️  ATENÇÃO: Preencha as API keys no arquivo .env antes de continuar!"
    echo ""
    echo "Chaves necessárias:"
    echo "  - OPENAI_API_KEY (https://platform.openai.com/api-keys)"
    echo "  - TAVILY_API_KEY (https://tavily.com/)"
    echo "  - OPENWEATHERMAP_API_KEY (https://openweathermap.org/api)"
    echo ""
    exit 1
fi

# 2. Verificar se as chaves estão preenchidas
if grep -q "OPENAI_API_KEY=$" .env || grep -q "TAVILY_API_KEY=$" .env; then
    echo "⚠️  Algumas API keys parecem estar vazias no .env"
    echo "   Verifique e preencha antes de executar docker compose up"
    echo ""
fi

echo "✅ Setup completo!"
echo ""
echo "Próximos passos:"
echo "  1. Verifique o arquivo .env e preencha as API keys"
echo "  2. Execute: docker compose up --build"
echo "  3. Acesse: http://localhost:8501"
echo ""
