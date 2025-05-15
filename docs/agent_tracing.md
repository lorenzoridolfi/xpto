# Agent Tracing and Root Cause Analysis

Este documento descreve o sistema de tracing e análise de root cause para agentes.

## Visão Geral

O sistema consiste em dois módulos principais:

1. `agent_tracer.py`: Rastreia eventos e mensagens dos agentes
2. `root_cause_analyzer.py`: Analisa os traces para identificar problemas e gerar recomendações

## Módulos

### AgentTracer

O `AgentTracer` é responsável por rastrear todas as interações dos agentes.

```python
from agent_tracer import AgentTracer

# Inicialização
tracer = AgentTracer(config)

# Durante a execução
tracer.on_messages_invoke(agent_name, messages)
tracer.on_messages_complete(agent_name, outputs)

# No final
tracer.save_trace("trace.json")
```

#### Estrutura de Eventos

Cada evento é representado pela classe `AgentEvent`:

```python
@dataclass
class AgentEvent:
    timestamp: str
    agent_name: str
    event_type: str
    inputs: List[Dict[str, str]]
    outputs: List[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]] = None
```

#### Configuração

O `AgentTracer` usa a mesma estrutura de configuração dos outros módulos:

```json
{
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file": "agent_tracer.log",
        "console": true
    }
}
```

### RootCauseAnalyzer

O `RootCauseAnalyzer` analisa os traces dos agentes para identificar problemas e gerar recomendações.

```python
from root_cause_analyzer import RootCauseAnalyzer

# Inicialização
analyzer = RootCauseAnalyzer(config)

# Análise
analysis = analyzer.analyze(tracer, user_feedback="O sistema está lento")
analyzer.save_analysis(analysis, "root_cause.json")
```

#### Tipos de Análise

O analisador identifica vários tipos de problemas:

1. **Performance**
   - Tempo de processamento dos agentes
   - Gargalos de comunicação
   - Uso de recursos

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

#### Estrutura da Análise

A análise é representada pela classe `RootCauseAnalysis`:

```python
@dataclass
class RootCauseAnalysis:
    summary: str
    issues: List[Dict[str, Any]]
    recommendations: List[str]
    metadata: Optional[Dict[str, Any]] = None
```

#### Configuração

O analisador usa configurações específicas para cada tipo de análise:

```json
{
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file": "root_cause.log",
        "console": true
    },
    "performance": {
        "response_time_threshold": 5.0,
        "error_rate_threshold": 0.1
    },
    "analysis": {
        "max_messages_per_agent": 10
    }
}
```

## Integração com Projetos

### Configuração do Projeto

1. Adicione as dependências:
   ```python
   from agent_tracer import AgentTracer
   from root_cause_analyzer import RootCauseAnalyzer
   ```

2. Configure os módulos:
   ```python
   # Carregue a configuração
   with open("config.json", "r") as f:
       config = json.load(f)
   
   # Inicialize os módulos
   tracer = AgentTracer(config)
   analyzer = RootCauseAnalyzer(config)
   ```

3. Integre com os agentes:
   ```python
   class MyAgent:
       def on_messages_invoke(self, messages):
           tracer.on_messages_invoke(self.name, messages)
   
       def on_messages_complete(self, messages):
           tracer.on_messages_complete(self.name, messages)
   ```

### Uso em Diferentes Cenários

1. **Debug e Desenvolvimento**
   ```python
   # Durante o desenvolvimento
   tracer.on_messages_invoke(agent_name, messages)
   tracer.on_messages_complete(agent_name, outputs)
   
   # Análise de problemas
   analysis = analyzer.analyze(tracer)
   print(analysis.summary)
   ```

2. **Produção**
   ```python
   # Trace completo
   tracer.save_trace("trace.json")
   
   # Análise com feedback
   analysis = analyzer.analyze(tracer, user_feedback="Problema reportado")
   analyzer.save_analysis(analysis, "root_cause.json")
   ```

3. **Monitoramento Contínuo**
   ```python
   # Trace periódico
   if time.time() - last_trace > TRACE_INTERVAL:
       tracer.save_trace(f"trace_{timestamp}.json")
   
   # Análise automática
   if has_issues:
       analysis = analyzer.analyze(tracer)
       notify_team(analysis)
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

3. **Configuração**
   - Ajuste thresholds conforme necessário
   - Configure logging apropriadamente
   - Mantenha configurações atualizadas

## Exemplos

### Trace Básico
```python
# Inicialização
tracer = AgentTracer(config)

# Durante a execução
tracer.on_messages_invoke("WriterAgent", messages)
tracer.on_messages_complete("WriterAgent", outputs)

# Salvar trace
tracer.save_trace("writer_trace.json")
```

### Análise Completa
```python
# Inicialização
analyzer = RootCauseAnalyzer(config)

# Análise com feedback
analysis = analyzer.analyze(
    tracer,
    user_feedback="O sistema está lento e com erros"
)

# Salvar resultados
analyzer.save_analysis(analysis, "analysis.json")

# Usar resultados
print(analysis.summary)
for issue in analysis.issues:
    print(f"- {issue['description']}")
for rec in analysis.recommendations:
    print(f"- {rec}")
``` 