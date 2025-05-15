# Guia de Logging para Desenvolvedores

## Visão Geral
Este guia explica o sistema de logging do projeto, projetado especificamente para desenvolvedores com experiência em programação mas novos em agentes de IA.

## Estrutura de Logging

### Categorias de Log
O sistema divide os logs em 4 categorias principais:

1. **Comunicação entre Agentes** (`agent_communication`)
   - Logs de mensagens entre agentes
   - Fluxo de dados e decisões
   - Formato: `[AGENT] Origem -> Destino: Mensagem`

2. **Estado dos Agentes** (`agent_state`)
   - Mudanças de estado dos agentes
   - Comportamento e transições
   - Formato: `[STATE] Agente: Estado - Detalhes`

3. **Performance** (`performance`)
   - Métricas de tempo de resposta
   - Uso de recursos
   - Formato: `[PERF] Agente: Métrica = Valor Unidade`

4. **Erros** (`error`)
   - Exceções e falhas
   - Stack traces
   - Formato: `[ERROR] Agente: Tipo - Mensagem\nStack: Trace`

## Arquivos de Log

- `toy_example.log`: Log principal
- `toy_example_agent_communication.log`: Logs de comunicação
- `toy_example_agent_state.log`: Logs de estado
- `toy_example_performance.log`: Logs de performance
- `toy_example_error.log`: Logs de erro

## Configuração

O sistema de logging é configurado em `logging_config.json`:

```json
{
    "logging": {
        "level": "DEBUG",
        "format": "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        "file": "toy_example.log",
        "console": true,
        "file_rotation": {
            "enabled": true,
            "max_size_mb": 10,
            "backup_count": 5
        }
    }
}
```

## Como Usar

### 1. Logging Básico
```python
import logging

logger = logging.getLogger("toy_example")
logger.info("Mensagem de informação")
logger.debug("Mensagem de debug")
logger.error("Mensagem de erro")
```

### 2. Logging de Agentes
```python
# Log de comunicação
comm_logger = logging.getLogger("toy_example.agent_communication")
comm_logger.info(f"Agent {agent_name} sending message to {target}")

# Log de estado
state_logger = logging.getLogger("toy_example.agent_state")
state_logger.debug(f"Agent {agent_name} state change", extra={
    "state": "processing",
    "details": {"input_size": len(inputs)}
})

# Log de performance
perf_logger = logging.getLogger("toy_example.performance")
perf_logger.info(f"Processing time", extra={
    "metric": "response_time",
    "value": 1.5,
    "unit": "seconds"
})

# Log de erro
error_logger = logging.getLogger("toy_example.error")
error_logger.error(f"Error in {agent_name}", extra={
    "error_type": "ValidationError",
    "message": str(error),
    "stack_trace": traceback.format_exc()
})
```

## Dicas para Debugging

1. **Comunicação entre Agentes**
   - Use `agent_communication` para entender o fluxo de dados
   - Procure por padrões de comunicação inesperados
   - Verifique se as mensagens estão no formato correto

2. **Estado dos Agentes**
   - Monitore `agent_state` para entender o comportamento
   - Identifique estados inesperados
   - Verifique transições de estado

3. **Performance**
   - Use `performance` para identificar gargalos
   - Monitore tempos de resposta
   - Verifique uso de recursos

4. **Erros**
   - Consulte `error` para debugging
   - Analise stack traces
   - Identifique padrões de erro

## Boas Práticas

1. **Níveis de Log**
   - DEBUG: Informações detalhadas para debugging
   - INFO: Informações gerais sobre operação
   - WARNING: Situações inesperadas mas não críticas
   - ERROR: Erros que precisam de atenção
   - CRITICAL: Erros que impedem operação

2. **Formatação**
   - Use mensagens claras e descritivas
   - Inclua contexto relevante
   - Mantenha logs concisos mas informativos

3. **Rotação de Arquivos**
   - Logs são rotacionados a cada 10MB
   - Mantém últimos 5 arquivos
   - Evita consumo excessivo de disco

## Troubleshooting

1. **Logs Muito Verbosos**
   - Ajuste nível para INFO ou WARNING
   - Desative categorias não essenciais
   - Use filtros específicos

2. **Logs Muito Escassos**
   - Aumente nível para DEBUG
   - Ative todas as categorias
   - Adicione mais pontos de logging

3. **Problemas de Performance**
   - Monitore tamanho dos arquivos
   - Ajuste frequência de rotação
   - Considere usar logging assíncrono

## Contribuindo

1. **Novos Logs**
   - Use categorias existentes quando possível
   - Adicione novas categorias se necessário
   - Documente novas categorias

2. **Modificações**
   - Mantenha formato consistente
   - Atualize documentação
   - Teste em diferentes níveis

3. **Sugestões**
   - Abra issues para melhorias
   - Proponha novos formatos
   - Compartilhe boas práticas 