#!/usr/bin/env python3

import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from agent_tracer import AgentTracer
from root_cause_analyzer import RootCauseAnalyzer

# ... existing code ...

def main():
    # Carregar configuração
    with open("config.json", "r") as f:
        config = json.load(f)
    
    # Inicializar tracer e analyzer
    tracer = AgentTracer(config)
    analyzer = RootCauseAnalyzer(config)
    
    try:
        # Inicializar agentes
        writer_agent = WriterAgent(config)
        editor_agent = EditorAgent(config)
        reviewer_agent = ReviewerAgent(config)
        
        # Processar manifesto
        tracer.on_messages_invoke("WriterAgent", [{"source": "user", "content": "Processar manifesto"}])
        writer_output = writer_agent.process_manifesto()
        tracer.on_messages_complete("WriterAgent", writer_output)
        
        tracer.on_messages_invoke("EditorAgent", writer_output)
        editor_output = editor_agent.edit_manifesto(writer_output)
        tracer.on_messages_complete("EditorAgent", editor_output)
        
        tracer.on_messages_invoke("ReviewerAgent", editor_output)
        reviewer_output = reviewer_agent.review_manifesto(editor_output)
        tracer.on_messages_complete("ReviewerAgent", reviewer_output)
        
        # Salvar trace
        tracer.save_trace("update_manifesto_trace.json")
        
        # Análise de root cause
        analysis = analyzer.analyze(tracer)
        analyzer.save_analysis(analysis, "update_manifesto_analysis.json")
        
        # Log dos resultados
        logging.info("Manifesto atualizado com sucesso")
        logging.info(f"Análise: {analysis.summary}")
        
    except Exception as e:
        # Trace do erro
        tracer.on_messages_invoke("Error", [{"source": "system", "content": str(e)}])
        tracer.save_trace("update_manifesto_error_trace.json")
        
        # Análise do erro
        analysis = analyzer.analyze(tracer, user_feedback=f"Erro: {str(e)}")
        analyzer.save_analysis(analysis, "update_manifesto_error_analysis.json")
        
        logging.error(f"Erro ao atualizar manifesto: {e}")
        raise

if __name__ == "__main__":
    main() 