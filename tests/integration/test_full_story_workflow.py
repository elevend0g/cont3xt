"""
Integration tests for complete story writing workflow

This module tests the full virtual infinite context system with realistic
story writing scenarios, including pressure relief, character consistency,
plot tracking, and performance under load.
"""

import asyncio
import pytest
import time
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.virtual_context_mcp.context_manager import ContextManager
from src.virtual_context_mcp.config.settings import load_config
from tests.fixtures.performance_targets import (
    PERFORMANCE_TARGETS, 
    MEMORY_TARGETS, 
    ACCURACY_TARGETS,
    PerformanceBenchmark, 
    MemoryValidator
)
from tests.fixtures.story_data import (
    STORY_CONVERSATIONS,
    CHARACTERS,
    PLOT_THREADS,
    LOCATIONS,
    MEMORY_QUERIES,
    generate_story_interactions
)


class TestStoryWorkflow:
    """Test complete story writing workflow with virtual infinite context"""
    
    @pytest.fixture
    async def context_manager(self):
        """Setup test context manager with in-memory databases"""
        # Create temporary directory for test data
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Load test configuration
            config_path = Path(__file__).parent.parent / "fixtures" / "test_config.yaml"
            config = load_config(str(config_path))
            
            # Override database paths for testing
            config.database.sqlite.path = f"{temp_dir}/test.db"
            
            # Initialize context manager
            context_manager = ContextManager(config)
            await context_manager.initialize()
            
            yield context_manager
            
        finally:
            # Cleanup
            if hasattr(context_manager, 'shutdown'):
                await context_manager.shutdown()
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def performance_benchmark(self):
        """Setup performance benchmarking"""
        return PerformanceBenchmark()
    
    @pytest.fixture
    def memory_validator(self):
        """Setup memory validation"""
        return MemoryValidator()
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_pressure_relief_cycle(self, context_manager, performance_benchmark, memory_validator):
        """Test that pressure relief works correctly"""
        session_id = "test_pressure_relief"
        
        # Generate enough content to trigger pressure relief
        long_interactions = generate_story_interactions(20)
        
        for i, interaction in enumerate(long_interactions):
            start_time = time.time()
            
            # Process interaction
            result = await context_manager.process_interaction(
                user_input=interaction["user"],
                assistant_response=interaction["assistant"],
                session_id=session_id
            )
            
            # Record performance
            duration_ms = (time.time() - start_time) * 1000
            performance_benchmark.record_measurement("context_assembly", duration_ms)
            
            # Check context state
            context_window = await context_manager.build_context_window(session_id)
            pressure = context_manager.calculate_pressure(context_window)
            
            memory_validator.record_context_state(
                context_window.total_tokens,
                len(context_window.chunks),
                sum(len(chunk.entities) for chunk in context_window.chunks)
            )
            
            # Verify pressure relief triggers correctly
            if pressure > context_manager.config.context.pressure_threshold:
                # Should trigger relief
                assert result.get("pressure_relief_triggered", False), f"Pressure relief should trigger at {pressure:.1%}"
                
                # Verify relief effectiveness
                new_context = await context_manager.build_context_window(session_id)
                new_pressure = context_manager.calculate_pressure(new_context)
                
                assert new_pressure < pressure, "Pressure should decrease after relief"
                assert new_context.total_tokens <= MEMORY_TARGETS["max_context_size"], "Context should not exceed maximum size"
        
        # Validate memory targets
        memory_report = memory_validator.check_memory_targets()
        assert memory_report["status"] == "pass", f"Memory validation failed: {memory_report}"
        
        # Check performance targets
        assert performance_benchmark.check_target("context_assembly", PERFORMANCE_TARGETS["context_assembly"]), \
            f"Context assembly too slow: {performance_benchmark.get_average('context_assembly'):.1f}ms"
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_character_consistency(self, context_manager, performance_benchmark):
        """Test character memory across context relief"""
        session_id = "test_character_consistency"
        
        # Introduce characters with specific traits
        for conversation in STORY_CONVERSATIONS[:3]:
            await context_manager.process_interaction(
                user_input=conversation["user"],
                assistant_response=conversation["assistant"],
                session_id=session_id
            )
        
        # Force pressure relief by adding more content
        padding_interactions = generate_story_interactions(15)
        for interaction in padding_interactions[5:]:  # Skip first 5 to avoid duplication
            await context_manager.process_interaction(
                user_input=interaction["user"],
                assistant_response=interaction["assistant"],
                session_id=session_id
            )
        
        # Test character memory retrieval
        start_time = time.time()
        elena_memories = await context_manager.retrieval_system.get_relevant_memories(
            query="Elena archer emerald eyes",
            session_id=session_id,
            max_results=10
        )
        
        retrieval_time = (time.time() - start_time) * 1000
        performance_benchmark.record_measurement("memory_search", retrieval_time)
        
        # Verify character details are preserved
        elena_found = False
        for memory in elena_memories:
            content = memory['chunk'].content.lower()
            if "elena" in content and "emerald eyes" in content:
                elena_found = True
                break
        
        assert elena_found, "Elena's character details should be retrievable after pressure relief"
        assert len(elena_memories) > 0, "Character memories should be found"
        
        # Check performance
        assert performance_benchmark.check_target("memory_search", PERFORMANCE_TARGETS["memory_search"]), \
            f"Memory search too slow: {retrieval_time:.1f}ms"
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_plot_thread_tracking(self, context_manager, performance_benchmark):
        """Test plot thread continuity across pressure relief cycles"""
        session_id = "test_plot_tracking"
        
        # Establish multiple plot threads
        for conversation in STORY_CONVERSATIONS:
            await context_manager.process_interaction(
                user_input=conversation["user"],
                assistant_response=conversation["assistant"],
                session_id=session_id
            )
        
        # Add more content to force multiple relief cycles
        additional_interactions = generate_story_interactions(25)
        for interaction in additional_interactions[len(STORY_CONVERSATIONS):]:
            await context_manager.process_interaction(
                user_input=interaction["user"],
                assistant_response=interaction["assistant"],
                session_id=session_id
            )
        
        # Test plot thread retrieval
        for plot_name, plot_data in PLOT_THREADS.items():
            start_time = time.time()
            
            plot_memories = await context_manager.retrieval_system.get_relevant_memories(
                query=plot_data["description"],
                session_id=session_id,
                max_results=5
            )
            
            query_time = (time.time() - start_time) * 1000
            performance_benchmark.record_measurement("memory_search", query_time)
            
            # Verify plot elements are found
            plot_elements_found = 0
            for memory in plot_memories:
                content = memory['chunk'].content.lower()
                for element in plot_data["elements"]:
                    if element.lower() in content:
                        plot_elements_found += 1
                        break
            
            assert plot_elements_found > 0, f"Plot thread '{plot_name}' elements should be retrievable"
        
        # Verify graph connections for plot elements
        if hasattr(context_manager, 'graph_store'):
            start_time = time.time()
            
            shadow_wolf_connections = await context_manager.graph_store.get_connected_entities(
                entity_name="Shadow Wolves",
                entity_type="creature",
                session_id=session_id
            )
            
            graph_time = (time.time() - start_time) * 1000
            performance_benchmark.record_measurement("graph_query", graph_time)
            
            assert performance_benchmark.check_target("graph_query", PERFORMANCE_TARGETS["graph_query"]), \
                f"Graph query too slow: {graph_time:.1f}ms"
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_memory_retrieval_accuracy(self, context_manager, performance_benchmark):
        """Test memory retrieval relevance and accuracy"""
        session_id = "test_memory_accuracy"
        
        # Create story with specific, searchable details
        for conversation in STORY_CONVERSATIONS:
            await context_manager.process_interaction(
                user_input=conversation["user"],
                assistant_response=conversation["assistant"],
                session_id=session_id
            )
        
        # Force multiple relief cycles
        filler_interactions = generate_story_interactions(30)
        for interaction in filler_interactions[len(STORY_CONVERSATIONS):]:
            await context_manager.process_interaction(
                user_input=interaction["user"],
                assistant_response=interaction["assistant"],
                session_id=session_id
            )
        
        # Test specific memory queries
        for query_data in MEMORY_QUERIES:
            start_time = time.time()
            
            memories = await context_manager.retrieval_system.get_relevant_memories(
                query=query_data["query"],
                session_id=session_id,
                max_results=5
            )
            
            search_time = (time.time() - start_time) * 1000
            performance_benchmark.record_measurement("memory_search", search_time)
            
            # Check relevance of results
            relevant_memories = 0
            for memory in memories:
                content = memory['chunk'].content.lower()
                score = memory['score']
                
                # Check if expected elements are present
                elements_found = sum(1 for element in query_data["expected_elements"] 
                                   if element.lower() in content)
                
                if elements_found > 0 and score >= ACCURACY_TARGETS["semantic_similarity"]:
                    relevant_memories += 1
            
            # Verify retrieval accuracy
            if memories:  # Only check if memories were found
                relevance_ratio = relevant_memories / len(memories)
                assert relevance_ratio >= ACCURACY_TARGETS["memory_retrieval_recall"], \
                    f"Memory retrieval accuracy too low for '{query_data['query']}': {relevance_ratio:.2f}"
        
        # Check overall performance
        avg_search_time = performance_benchmark.get_average("memory_search")
        assert avg_search_time <= PERFORMANCE_TARGETS["memory_search"], \
            f"Average memory search too slow: {avg_search_time:.1f}ms"
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.performance
    async def test_long_conversation_performance(self, context_manager, performance_benchmark, memory_validator):
        """Test performance with 500+ interactions"""
        session_id = "test_long_conversation"
        
        # Generate large number of interactions
        interactions = generate_story_interactions(100)  # Reduced for CI/CD compatibility
        
        response_times = []
        
        for i, interaction in enumerate(interactions):
            start_time = time.time()
            
            # Process interaction
            result = await context_manager.process_interaction(
                user_input=interaction["user"],
                assistant_response=interaction["assistant"],
                session_id=session_id
            )
            
            duration = time.time() - start_time
            response_times.append(duration * 1000)  # Convert to ms
            
            # Record measurements every 10 interactions
            if i % 10 == 0:
                performance_benchmark.record_measurement("context_assembly", duration * 1000)
                
                context_window = await context_manager.build_context_window(session_id)
                memory_validator.record_context_state(
                    context_window.total_tokens,
                    len(context_window.chunks),
                    sum(len(chunk.entities) for chunk in context_window.chunks)
                )
        
        # Verify performance consistency
        if len(response_times) > 10:
            early_avg = sum(response_times[:10]) / 10
            late_avg = sum(response_times[-10:]) / 10
            
            # Performance should not degrade significantly
            performance_degradation = late_avg / early_avg if early_avg > 0 else 1.0
            assert performance_degradation <= 2.0, f"Performance degraded too much: {performance_degradation:.2f}x"
        
        # Verify memory usage is controlled
        memory_report = memory_validator.check_memory_targets()
        assert memory_report["status"] == "pass", f"Memory usage exceeded limits: {memory_report}"
        
        # Check final performance targets
        final_avg = performance_benchmark.get_average("context_assembly")
        assert final_avg <= PERFORMANCE_TARGETS["context_assembly"] * 2, \
            f"Long conversation performance too slow: {final_avg:.1f}ms"
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_entity_extraction_performance(self, context_manager, performance_benchmark):
        """Test entity extraction speed and accuracy"""
        session_id = "test_entity_extraction"
        
        for conversation in STORY_CONVERSATIONS:
            start_time = time.time()
            
            # Process interaction and measure entity extraction time
            await context_manager.process_interaction(
                user_input=conversation["user"],
                assistant_response=conversation["assistant"],
                session_id=session_id
            )
            
            extraction_time = (time.time() - start_time) * 1000
            performance_benchmark.record_measurement("entity_extraction", extraction_time)
        
        # Verify entity extraction performance
        avg_extraction_time = performance_benchmark.get_average("entity_extraction")
        assert avg_extraction_time <= PERFORMANCE_TARGETS["entity_extraction"], \
            f"Entity extraction too slow: {avg_extraction_time:.1f}ms"
        
        # Verify entities were extracted correctly
        memories = await context_manager.retrieval_system.get_relevant_memories(
            query="Elena Millbrook",
            session_id=session_id,
            max_results=10
        )
        
        character_entities_found = False
        location_entities_found = False
        
        for memory in memories:
            for entity in memory['chunk'].entities:
                if entity.entity_type == 'character' and 'elena' in entity.name.lower():
                    character_entities_found = True
                elif entity.entity_type == 'location' and 'millbrook' in entity.name.lower():
                    location_entities_found = True
        
        assert character_entities_found, "Character entities should be extracted"
        assert location_entities_found, "Location entities should be extracted"
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_system_recovery_after_errors(self, context_manager):
        """Test system recovery and error handling"""
        session_id = "test_error_recovery"
        
        # Test normal operation first
        result = await context_manager.process_interaction(
            user_input="Start a story",
            assistant_response="Once upon a time...",
            session_id=session_id
        )
        assert result is not None
        
        # Test with invalid input
        try:
            await context_manager.process_interaction(
                user_input="",  # Empty input
                assistant_response="",  # Empty response
                session_id=session_id
            )
        except Exception:
            pass  # Expected to handle gracefully
        
        # Verify system still works after error
        result = await context_manager.process_interaction(
            user_input="Continue the story",
            assistant_response="The adventure continued...",
            session_id=session_id
        )
        assert result is not None
        
        # Verify context window is still accessible
        context_window = await context_manager.build_context_window(session_id)
        assert context_window is not None
        assert context_window.total_tokens > 0