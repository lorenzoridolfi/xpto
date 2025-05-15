# Agent Tracing and Root Cause Analysis

Este documento descreve o sistema de tracing, análise de root cause e analytics para agentes.

## Visão Geral

O sistema consiste em três módulos principais:

1. `agent_tracer.py`: Rastreia eventos e mensagens dos agentes
2. `root_cause_analyzer.py`: Analisa os traces para identificar problemas
3. `tool_analytics.py`: Analisa o uso de ferramentas pelos agentes

## Módulos

### 1. AgentTracer

O `AgentTracer` é responsável por rastrear todas as interações dos agentes.

#### Estrutura de Dados

```python
@dataclass
class TokenUsage:
    """Estatísticas de uso de tokens do LLM."""
    prompt_tokens: int      # Tokens usados no prompt
    completion_tokens: int  # Tokens usados na resposta
    total_tokens: int       # Total de tokens
    model: str             # Modelo usado (ex: gpt-4)

@dataclass
class AgentEvent:
    """Evento do agente."""
    timestamp: str                    # Data/hora do evento
    agent_name: str                   # Nome do agente
    event_type: str                   # Tipo (invoke/complete)
    inputs: List[Dict[str, str]]      # Mensagens de entrada
    outputs: List[Dict[str, Any]]     # Mensagens de saída
    metadata: Optional[Dict[str, Any]] # Metadados extras
    token_usage: Optional[TokenUsage]  # Uso de tokens
```

#### Funcionalidades

1. **Tracing de Eventos**
   ```python
   # Início do processamento
   tracer.on_messages_invoke("WriterAgent", messages, token_usage)
   
   # Fim do processamento
   tracer.on_messages_complete("WriterAgent", outputs, token_usage)
   ```

2. **Tracing de Tokens**
   - Registra tokens de prompt e completion
   - Identifica o modelo usado
   - Calcula total de tokens

3. **Tracing de Tempo**
   - Tempo de início
   - Tempo de processamento
   - Tempo de fim

4. **Persistência**
   - Salva traces em JSON
   - Mantém histórico de eventos
   - Permite análise posterior

### 2. RootCauseAnalyzer

O `RootCauseAnalyzer` analisa os traces para identificar problemas e gerar recomendações.

#### Estrutura de Dados

```python
@dataclass
class RootCauseAnalysis:
    """Resultado da análise de root cause."""
    summary: str                    # Resumo da análise
    issues: List[Dict[str, Any]]    # Problemas encontrados
    recommendations: List[str]      # Recomendações
    metadata: Optional[Dict[str, Any]] # Metadados extras
```

#### Tipos de Análise

1. **Performance**
   - Tempo de processamento
   - Uso de tokens
   - Gargalos de comunicação

2. **Comunicação**
   - Padrões de mensagens
   - Volume de comunicação
   - Eficiência do fluxo

3. **Erros**
   - Exceções e erros
   - Falhas de validação
   - Problemas de integração

4. **Feedback do Usuário**
   - Problemas reportados
   - Sugestões de melhoria
   - Experiência do usuário

#### Funcionalidades

1. **Análise de Eventos**
   ```python
   # Análise completa
   analysis = analyzer.analyze(tracer)
   
   # Análise com feedback
   analysis = analyzer.analyze(tracer, user_feedback="Sistema lento")
   ```

2. **Geração de Recomendações**
   - Baseadas em problemas encontrados
   - Considerando feedback do usuário
   - Priorizadas por severidade

3. **Persistência**
   - Salva análises em JSON
   - Mantém histórico de problemas
   - Permite tracking de melhorias

### 3. ToolAnalytics

O `ToolAnalytics` analisa o uso de ferramentas pelos agentes.

#### Estrutura de Dados

```python
@dataclass
class ToolUsage:
    """Uso de uma ferramenta."""
    tool_name: str                  # Nome da ferramenta
    call_count: int                 # Número de chamadas
    success_rate: float             # Taxa de sucesso
    avg_duration: float             # Duração média
    error_count: int                # Número de erros
    last_used: str                  # Último uso
```

#### Tipos de Análise

1. **Uso de Ferramentas**
   - Frequência de uso
   - Taxa de sucesso
   - Tempo de execução

2. **Padrões de Uso**
   - Sequência de ferramentas
   - Combinações comuns
   - Dependências

3. **Problemas**
   - Erros frequentes
   - Timeouts
   - Falhas de integração

#### Funcionalidades

1. **Análise de Uso**
   ```python
   # Análise de ferramenta
   usage = analytics.analyze_tool("search_tool")
   
   # Análise de agente
   usage = analytics.analyze_agent("WriterAgent")
   ```

2. **Recomendações**
   - Otimização de uso
   - Substituição de ferramentas
   - Melhorias de integração

3. **Persistência**
   - Salva analytics em JSON
   - Mantém histórico de uso
   - Permite análise de tendências

## Integração

### Configuração

```json
{
  "logging": {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "agent_tracer.log",
    "console": true
  },
  "analysis": {
    "performance": {
      "response_time_threshold": 5.0,
      "error_rate_threshold": 0.1
    },
    "token_tracking": {
      "enabled": true
    }
  }
}
```

### Uso em Projetos

1. **Inicialização**
   ```python
   # Carregar configuração
   with open("config.json", "r") as f:
       config = json.load(f)
   
   # Inicializar módulos
   tracer = AgentTracer(config)
   analyzer = RootCauseAnalyzer(config)
   analytics = ToolAnalytics(config)
   ```

2. **Tracing**
   ```python
   # Durante a execução
   tracer.on_messages_invoke(agent_name, messages, token_usage)
   tracer.on_messages_complete(agent_name, outputs, token_usage)
   ```

3. **Análise**
   ```python
   # Análise de root cause
   analysis = analyzer.analyze(tracer)
   
   # Análise de ferramentas
   tool_analysis = analytics.analyze_tool("search_tool")
   ```

4. **Persistência**
   ```python
   # Salvar traces
   tracer.save_trace("trace.json")
   
   # Salvar análises
   analyzer.save_analysis(analysis, "analysis.json")
   analytics.save_analysis(tool_analysis, "tool_analysis.json")
   ```

## Boas Práticas

1. **Tracing**
   - Trace todos os eventos importantes
   - Inclua metadata relevante
   - Mantenha os traces organizados

2. **Análise**
   - Analise traces regularmente
   - Considere feedback dos usuários
   - Implemente recomendações

3. **Ferramentas**
   - Monitore uso de ferramentas
   - Otimize padrões de uso
   - Mantenha histórico de problemas

4. **Configuração**
   - Ajuste thresholds conforme necessário
   - Configure logging apropriadamente
   - Mantenha configurações atualizadas

## Exemplos

### Trace Completo
```python
# Inicialização
tracer = AgentTracer(config)
analyzer = RootCauseAnalyzer(config)
analytics = ToolAnalytics(config)

# Durante a execução
tracer.on_messages_invoke("WriterAgent", messages, token_usage)
writer_output = writer_agent.process_text()
tracer.on_messages_complete("WriterAgent", writer_output, token_usage)

# Análise
analysis = analyzer.analyze(tracer)
tool_analysis = analytics.analyze_tool("search_tool")

# Salvar resultados
tracer.save_trace("trace.json")
analyzer.save_analysis(analysis, "analysis.json")
analytics.save_analysis(tool_analysis, "tool_analysis.json")
```

### Análise de Erro
```python
try:
    # Processamento normal
    tracer.on_messages_invoke("WriterAgent", messages)
    writer_output = writer_agent.process_text()
    tracer.on_messages_complete("WriterAgent", writer_output)
except Exception as e:
    # Trace do erro
    tracer.on_messages_invoke("Error", [{"source": "system", "content": str(e)}])
    tracer.save_trace("error_trace.json")
    
    # Análise do erro
    analysis = analyzer.analyze(tracer, user_feedback=f"Erro: {str(e)}")
    analyzer.save_analysis(analysis, "error_analysis.json")
    
    # Análise de ferramentas
    tool_analysis = analytics.analyze_tool("search_tool")
    analytics.save_analysis(tool_analysis, "error_tool_analysis.json")
``` 