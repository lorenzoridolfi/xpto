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

# Agent Tracing e Integração com Autogen

## Visão Geral

O `AgentTracer` é um componente de observabilidade que se integra com os agentes do Autogen para fornecer:
- Rastreamento de eventos
- Métricas de uso de tokens
- Estatísticas de cache
- Logs detalhados
- Análise de performance

## Arquitetura

```
Autogen Agents (AssistantAgent, UserProxyAgent)
        ↓
AgentTracer (Observador)
        ↓
Logs, Métricas, Traces
```

### Componentes

1. **Agentes Autogen**
   - `AssistantAgent`: Agente principal que processa mensagens
   - `UserProxyAgent`: Interface com o usuário
   - `GroupChat`: Coordenação entre agentes

2. **AgentTracer**
   - Observador dos agentes
   - Coletor de métricas
   - Gerador de logs
   - Calculador de estatísticas

3. **Sistema de Logging**
   - Arquivos de log
   - Console output
   - Formatação personalizada

## Integração

### 1. Inicialização

```python
from autogen import AssistantAgent, UserProxyAgent
from agent_tracer import AgentTracer, TokenUsage

# Configuração do tracer
config = {
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file": "agent_trace.log",
        "console": true
    }
}

# Criar tracer
tracer = AgentTracer(config)

# Criar agentes
assistant = AssistantAgent(
    name="assistant",
    system_message="You are a helpful assistant.",
    llm_config={"config_list": [{"model": "gpt-4"}]}
)

user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="TERMINATE"
)
```

### 2. Tracing de Eventos

```python
# Antes de processar mensagens
tracer.on_messages_invoke(
    agent_name="assistant",
    messages=[{"role": "user", "content": "Hello"}],
    token_usage=TokenUsage(
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        model="gpt-4"
    ),
    cache_hit=False
)

# Após processar mensagens
tracer.on_messages_complete(
    agent_name="assistant",
    outputs=[{"role": "assistant", "content": "Hi there!"}],
    token_usage=TokenUsage(
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        model="gpt-4"
    ),
    cache_hit=True
)
```

### 3. Integração com Cache

```python
from llm_cache import LLMCache

# Inicializar cache
cache = LLMCache(
    max_size=1000,
    similarity_threshold=0.85,
    expiration_hours=24
)

# Verificar cache antes de chamar LLM
cache_key = generate_cache_key(messages)
cached_response = cache.get(cache_key)

if cached_response:
    # Cache hit
    tracer.on_messages_complete(
        agent_name="assistant",
        outputs=cached_response,
        token_usage=TokenUsage(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            model="gpt-4"
        ),
        cache_hit=True,
        cache_key=cache_key
    )
else:
    # Cache miss
    response = assistant.generate_response(messages)
    cache.set(cache_key, response)
    
    tracer.on_messages_complete(
        agent_name="assistant",
        outputs=response,
        token_usage=TokenUsage(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            model="gpt-4"
        ),
        cache_hit=False
    )
```

### 4. Análise de Performance

```python
# Obter estatísticas de cache
cache_stats = tracer.get_cache_statistics()
print(f"Cache hit rate: {cache_stats['savings_percentage']}%")
print(f"Total tokens saved: {cache_stats['total_savings']}")

# Salvar trace completo
tracer.save_trace("agent_trace.json")
```

## Exemplo Completo

```python
from autogen import AssistantAgent, UserProxyAgent, GroupChat
from agent_tracer import AgentTracer, TokenUsage
from llm_cache import LLMCache

def setup_agents_with_tracing():
    # Configuração
    config = {
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file": "agent_trace.log",
            "console": true
        }
    }
    
    # Inicializar componentes
    tracer = AgentTracer(config)
    cache = LLMCache(max_size=1000)
    
    # Criar agentes
    assistant = AssistantAgent(
        name="assistant",
        system_message="You are a helpful assistant.",
        llm_config={"config_list": [{"model": "gpt-4"}]}
    )
    
    user_proxy = UserProxyAgent(
        name="user_proxy",
        human_input_mode="TERMINATE"
    )
    
    # Criar group chat
    groupchat = GroupChat(
        agents=[user_proxy, assistant],
        messages=[],
        max_round=10
    )
    
    # Função wrapper para tracing
    def traced_chat(agent_name, messages):
        # Trace início
        tracer.on_messages_invoke(agent_name, messages)
        
        # Verificar cache
        cache_key = generate_cache_key(messages)
        cached_response = cache.get(cache_key)
        
        if cached_response:
            # Cache hit
            tracer.on_messages_complete(
                agent_name,
                cached_response,
                token_usage=TokenUsage(
                    prompt_tokens=10,
                    completion_tokens=20,
                    total_tokens=30,
                    model="gpt-4"
                ),
                cache_hit=True,
                cache_key=cache_key
            )
            return cached_response
        
        # Cache miss - processar normalmente
        response = assistant.generate_response(messages)
        cache.set(cache_key, response)
        
        # Trace fim
        tracer.on_messages_complete(
            agent_name,
            response,
            token_usage=TokenUsage(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                model="gpt-4"
            ),
            cache_hit=False
        )
        
        return response
    
    return {
        "tracer": tracer,
        "cache": cache,
        "assistant": assistant,
        "user_proxy": user_proxy,
        "groupchat": groupchat,
        "traced_chat": traced_chat
    }

# Uso
def main():
    # Setup
    components = setup_agents_with_tracing()
    
    # Iniciar chat
    components["user_proxy"].initiate_chat(
        components["groupchat"],
        message="Hello, how can you help me?"
    )
    
    # Análise final
    cache_stats = components["tracer"].get_cache_statistics()
    print(f"Cache hit rate: {cache_stats['savings_percentage']}%")
    print(f"Total tokens saved: {cache_stats['total_savings']}")
    
    # Salvar trace
    components["tracer"].save_trace("chat_trace.json")
```

## Boas Práticas

1. **Configuração**
   - Configure o logging apropriadamente
   - Ajuste os níveis de log conforme necessário
   - Use formatos de log consistentes

2. **Tracing**
   - Trace todos os eventos importantes
   - Inclua metadata relevante
   - Mantenha os traces organizados

3. **Cache**
   - Use chaves de cache consistentes
   - Monitore hit rates
   - Ajuste thresholds conforme necessário

4. **Performance**
   - Monitore uso de tokens
   - Acompanhe tempos de resposta
   - Analise padrões de uso

5. **Manutenção**
   - Limpe traces antigos
   - Rotacione logs
   - Mantenha estatísticas atualizadas

## Troubleshooting

1. **Logs não aparecem**
   - Verifique configuração de logging
   - Confirme níveis de log
   - Verifique permissões de arquivo

2. **Cache não está funcionando**
   - Verifique chaves de cache
   - Confirme thresholds
   - Monitore hit rates

3. **Performance ruim**
   - Analise traces
   - Verifique uso de tokens
   - Otimize configurações

4. **Erros de integração**
   - Verifique ordem de chamadas
   - Confirme tipos de dados
   - Valide configurações 