"""
Poetics Platform Integration Layer

Integrates all four neurosymbolic AI systems with the Poetics VR venue management platform.
Provides unified API for XR applications.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable
import asyncio
import json

import numpy as np
from supabase import create_client, Client

from ..systems.spatial_reasoning import SpatialReasoningEngine, SpatialEntity
from ..systems.hybrid_slam import HybridSLAM, Pose
from ..systems.training_assistant import TrainingAssistant, TrainingProcedure, HandPose, GazePose
from ..systems.asset_generator import AssetGenerator, AssetSpecification, GeneratedAsset


@dataclass
class VenueContext:
    """Context information for a Poetics venue."""
    
    venue_id: str
    venue_name: str
    venue_type: str  # 'theater', 'gallery', 'conference', etc.
    capacity: int
    dimensions: Dict[str, float]  # width, height, depth
    features: List[str]  # Available features
    metadata: Dict[str, any]
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            'venue_id': self.venue_id,
            'venue_name': self.venue_name,
            'venue_type': self.venue_type,
            'capacity': self.capacity,
            'dimensions': self.dimensions,
            'features': self.features,
            'metadata': self.metadata
        }


@dataclass
class SessionContext:
    """Context for an active XR session."""
    
    session_id: str
    venue_id: str
    user_id: str
    start_time: float
    mode: str  # 'exploration', 'training', 'performance', 'design'
    active_systems: List[str]  # Which neurosymbolic systems are active
    state: Dict[str, any]
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            'session_id': self.session_id,
            'venue_id': self.venue_id,
            'user_id': self.user_id,
            'start_time': self.start_time,
            'mode': self.mode,
            'active_systems': self.active_systems,
            'state': self.state
        }


class PoeticsIntegration:
    """
    Integration layer for Poetics VR platform.
    
    Provides unified interface to all four neurosymbolic AI systems:
    1. Spatial Reasoning Engine - Scene understanding
    2. Hybrid SLAM - Tracking and localization
    3. Training Assistant - Interactive training
    4. Asset Generator - Content creation
    
    Args:
        supabase_url: Supabase project URL
        supabase_key: Supabase API key
        neural_config: Configuration for neural components
        device: Computation device
    """
    
    def __init__(
        self,
        supabase_url: str,
        supabase_key: str,
        neural_config: Dict,
        device: str = "cuda"
    ):
        self.device = device
        
        # Initialize Supabase client
        self.supabase: Client = create_client(supabase_url, supabase_key)
        
        # Initialize neurosymbolic systems
        self.spatial_reasoning = SpatialReasoningEngine(
            gaussian_config=neural_config.get('gaussian', {}),
            segmentation_config=neural_config.get('segmentation', {}),
            grounding_config=neural_config.get('grounding', {}),
            device=device
        )
        
        self.slam = HybridSLAM(
            vocab_path=neural_config.get('slam', {}).get('vocab_path', 'ORBvoc.txt'),
            config_path=neural_config.get('slam', {}).get('config_path', 'config.yaml'),
            use_imu=neural_config.get('slam', {}).get('use_imu', True),
            device=device
        )
        
        self.training_assistant = TrainingAssistant(
            device=device,
            use_prolog=neural_config.get('training', {}).get('use_prolog', True)
        )
        
        self.asset_generator = AssetGenerator(
            model_name=neural_config.get('asset_gen', {}).get('model_name', 'stabilityai/stable-diffusion-2-1'),
            device=device,
            use_constraints=neural_config.get('asset_gen', {}).get('use_constraints', True)
        )
        
        # Session management
        self.active_sessions: Dict[str, SessionContext] = {}
        self.venue_cache: Dict[str, VenueContext] = {}
        
        # Event callbacks
        self.event_callbacks: Dict[str, List[Callable]] = {
            'scene_updated': [],
            'pose_updated': [],
            'training_feedback': [],
            'asset_generated': []
        }
    
    # Venue Management
    
    async def load_venue(self, venue_id: str) -> VenueContext:
        """
        Load venue information from Poetics database.
        
        Args:
            venue_id: Venue identifier
            
        Returns:
            VenueContext
        """
        # Check cache
        if venue_id in self.venue_cache:
            return self.venue_cache[venue_id]
        
        # Query Supabase
        response = self.supabase.table('venues').select('*').eq('id', venue_id).execute()
        
        if not response.data:
            raise ValueError(f"Venue not found: {venue_id}")
        
        venue_data = response.data[0]
        
        venue = VenueContext(
            venue_id=venue_data['id'],
            venue_name=venue_data['name'],
            venue_type=venue_data.get('type', 'generic'),
            capacity=venue_data.get('capacity', 100),
            dimensions=venue_data.get('dimensions', {'width': 10, 'height': 3, 'depth': 10}),
            features=venue_data.get('features', []),
            metadata=venue_data.get('metadata', {})
        )
        
        # Cache venue
        self.venue_cache[venue_id] = venue
        
        return venue
    
    async def save_venue_state(self, venue_id: str, state: Dict) -> None:
        """
        Save venue state to database.
        
        Args:
            venue_id: Venue identifier
            state: State dictionary to save
        """
        # Update venue state in Supabase
        self.supabase.table('venue_states').upsert({
            'venue_id': venue_id,
            'state': state,
            'updated_at': 'now()'
        }).execute()
    
    # Session Management
    
    async def start_session(
        self,
        venue_id: str,
        user_id: str,
        mode: str = 'exploration'
    ) -> SessionContext:
        """
        Start a new XR session.
        
        Args:
            venue_id: Venue identifier
            user_id: User identifier
            mode: Session mode
            
        Returns:
            SessionContext
        """
        import time
        import uuid
        
        session_id = str(uuid.uuid4())
        
        # Determine active systems based on mode
        active_systems = []
        if mode in ['exploration', 'performance']:
            active_systems.extend(['spatial_reasoning', 'slam'])
        if mode == 'training':
            active_systems.extend(['training_assistant', 'slam'])
        if mode == 'design':
            active_systems.extend(['spatial_reasoning', 'asset_generator'])
        
        session = SessionContext(
            session_id=session_id,
            venue_id=venue_id,
            user_id=user_id,
            start_time=time.time(),
            mode=mode,
            active_systems=active_systems,
            state={}
        )
        
        self.active_sessions[session_id] = session
        
        # Load venue
        venue = await self.load_venue(venue_id)
        
        # Initialize systems
        if 'slam' in active_systems:
            # Try to load existing map
            map_path = f"maps/{venue_id}"
            self.slam.load_map(map_path)
        
        # Log session start
        self.supabase.table('sessions').insert({
            'id': session_id,
            'venue_id': venue_id,
            'user_id': user_id,
            'mode': mode,
            'start_time': session.start_time
        }).execute()
        
        return session
    
    async def end_session(self, session_id: str) -> Dict:
        """
        End XR session and save data.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session summary
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Session not found: {session_id}")
        
        session = self.active_sessions[session_id]
        
        import time
        end_time = time.time()
        duration = end_time - session.start_time
        
        # Save SLAM map if active
        if 'slam' in session.active_systems:
            map_path = f"maps/{session.venue_id}"
            self.slam.save_map(map_path)
        
        # Save spatial scene graph if active
        if 'spatial_reasoning' in session.active_systems:
            scene_data = {
                'entities': [ent.to_dict() for ent in self.spatial_reasoning.entities.values()],
                'relations': [rel.to_triple() for rel in self.spatial_reasoning.relations]
            }
            await self.save_venue_state(session.venue_id, scene_data)
        
        # Update session in database
        self.supabase.table('sessions').update({
            'end_time': end_time,
            'duration': duration,
            'state': session.state
        }).eq('id', session_id).execute()
        
        # Remove from active sessions
        del self.active_sessions[session_id]
        
        return {
            'session_id': session_id,
            'duration': duration,
            'mode': session.mode
        }
    
    # Spatial Reasoning Integration
    
    async def capture_venue_scene(
        self,
        session_id: str,
        images: List[np.ndarray],
        camera_poses: List[np.ndarray],
        camera_intrinsics: np.ndarray
    ) -> Dict:
        """
        Capture and understand venue scene.
        
        Args:
            session_id: Session identifier
            images: List of RGB images
            camera_poses: List of camera poses
            camera_intrinsics: Camera intrinsic matrix
            
        Returns:
            Scene understanding results
        """
        session = self.active_sessions.get(session_id)
        if not session or 'spatial_reasoning' not in session.active_systems:
            raise ValueError("Spatial reasoning not active for this session")
        
        # Capture scene
        symbolic_scene = self.spatial_reasoning.capture_and_understand_scene(
            images=images,
            camera_poses=camera_poses,
            camera_intrinsics=camera_intrinsics,
            num_iterations=5000  # Reduced for real-time
        )
        
        # Emit event
        await self._emit_event('scene_updated', {
            'session_id': session_id,
            'num_entities': len(symbolic_scene.entities),
            'num_relations': len(symbolic_scene.relations)
        })
        
        return {
            'entities': symbolic_scene.entities,
            'relations': symbolic_scene.relations,
            'properties': symbolic_scene.properties
        }
    
    async def query_venue_scene(self, session_id: str, query: str) -> List[Dict]:
        """
        Query venue scene with natural language.
        
        Args:
            session_id: Session identifier
            query: Natural language query
            
        Returns:
            Query results
        """
        session = self.active_sessions.get(session_id)
        if not session or 'spatial_reasoning' not in session.active_systems:
            raise ValueError("Spatial reasoning not active for this session")
        
        results = self.spatial_reasoning.query_scene(query)
        
        return results
    
    # SLAM Integration
    
    async def track_user_pose(
        self,
        session_id: str,
        image: np.ndarray,
        timestamp: float,
        imu_data: Optional[List] = None
    ) -> Optional[Pose]:
        """
        Track user pose in venue.
        
        Args:
            session_id: Session identifier
            image: RGB image
            timestamp: Frame timestamp
            imu_data: Optional IMU measurements
            
        Returns:
            Pose or None if tracking lost
        """
        session = self.active_sessions.get(session_id)
        if not session or 'slam' not in session.active_systems:
            raise ValueError("SLAM not active for this session")
        
        # Track pose
        pose = self.slam.track_frame(image, timestamp, imu_data)
        
        if pose:
            # Update session state
            session.state['last_pose'] = pose.to_dict()
            
            # Emit event
            await self._emit_event('pose_updated', {
                'session_id': session_id,
                'pose': pose.to_dict()
            })
        
        return pose
    
    async def get_venue_map(self, session_id: str) -> Dict:
        """
        Get current SLAM map for venue.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Map data
        """
        session = self.active_sessions.get(session_id)
        if not session or 'slam' not in session.active_systems:
            raise ValueError("SLAM not active for this session")
        
        landmarks = self.slam.get_map_points()
        trajectory = self.slam.get_trajectory()
        semantic_memory = list(self.slam.semantic_memory.values())
        
        return {
            'landmarks': [lm.to_dict() for lm in landmarks],
            'trajectory': [pose.to_dict() for pose in trajectory],
            'semantic_memory': [mem.to_dict() for mem in semantic_memory]
        }
    
    # Training Integration
    
    async def start_training_procedure(
        self,
        session_id: str,
        procedure: TrainingProcedure
    ) -> None:
        """
        Start training procedure in session.
        
        Args:
            session_id: Session identifier
            procedure: Training procedure to execute
        """
        session = self.active_sessions.get(session_id)
        if not session or 'training_assistant' not in session.active_systems:
            raise ValueError("Training assistant not active for this session")
        
        self.training_assistant.start_procedure(procedure)
        
        # Register feedback callback
        def feedback_handler(feedback):
            asyncio.create_task(self._emit_event('training_feedback', {
                'session_id': session_id,
                'feedback': feedback
            }))
        
        self.training_assistant.add_feedback_callback(feedback_handler)
    
    async def update_training(
        self,
        session_id: str,
        left_hand: Optional[HandPose] = None,
        right_hand: Optional[HandPose] = None,
        gaze: Optional[GazePose] = None,
        world_state: Optional[Dict] = None
    ) -> Dict:
        """
        Update training assistant with tracking data.
        
        Args:
            session_id: Session identifier
            left_hand: Left hand pose
            right_hand: Right hand pose
            gaze: Gaze pose
            world_state: World state facts
            
        Returns:
            Training update results
        """
        session = self.active_sessions.get(session_id)
        if not session or 'training_assistant' not in session.active_systems:
            raise ValueError("Training assistant not active for this session")
        
        result = self.training_assistant.update(
            left_hand=left_hand,
            right_hand=right_hand,
            gaze=gaze,
            world_state=world_state
        )
        
        return result
    
    # Asset Generation Integration
    
    async def generate_venue_asset(
        self,
        session_id: str,
        specification: AssetSpecification
    ) -> GeneratedAsset:
        """
        Generate asset for venue.
        
        Args:
            session_id: Session identifier
            specification: Asset specification
            
        Returns:
            Generated asset
        """
        session = self.active_sessions.get(session_id)
        if not session or 'asset_generator' not in session.active_systems:
            raise ValueError("Asset generator not active for this session")
        
        # Generate asset
        asset = self.asset_generator.generate_asset(specification)
        
        # Save to database
        self.supabase.table('generated_assets').insert({
            'id': asset.asset_id,
            'session_id': session_id,
            'venue_id': session.venue_id,
            'specification': specification.to_dict(),
            'asset_type': asset.asset_type.value,
            'generation_time': asset.generation_time
        }).execute()
        
        # Emit event
        await self._emit_event('asset_generated', {
            'session_id': session_id,
            'asset_id': asset.asset_id,
            'asset_type': asset.asset_type.value
        })
        
        return asset
    
    # Event System
    
    def on(self, event_type: str, callback: Callable) -> None:
        """
        Register event callback.
        
        Args:
            event_type: Event type to listen for
            callback: Callback function
        """
        if event_type in self.event_callbacks:
            self.event_callbacks[event_type].append(callback)
    
    async def _emit_event(self, event_type: str, data: Dict) -> None:
        """
        Emit event to registered callbacks.
        
        Args:
            event_type: Event type
            data: Event data
        """
        if event_type in self.event_callbacks:
            for callback in self.event_callbacks[event_type]:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
    
    # Utility Methods
    
    def get_session_status(self, session_id: str) -> Dict:
        """Get current session status."""
        session = self.active_sessions.get(session_id)
        if not session:
            return {'status': 'not_found'}
        
        import time
        return {
            'status': 'active',
            'session': session.to_dict(),
            'duration': time.time() - session.start_time,
            'systems': {
                'spatial_reasoning': {
                    'entities': len(self.spatial_reasoning.entities),
                    'relations': len(self.spatial_reasoning.relations)
                } if 'spatial_reasoning' in session.active_systems else None,
                'slam': {
                    'tracking': not self.slam.lost_tracking,
                    'landmarks': len(self.slam.landmarks)
                } if 'slam' in session.active_systems else None,
                'training': {
                    'procedure_active': self.training_assistant.current_procedure is not None
                } if 'training_assistant' in session.active_systems else None
            }
        }


def create_poetics_integration(config: Dict) -> PoeticsIntegration:
    """
    Factory function to create Poetics integration.
    
    Args:
        config: Configuration dict with keys:
            - supabase_url: str
            - supabase_key: str
            - neural_config: dict
            - device: str
            
    Returns:
        Configured PoeticsIntegration
    """
    return PoeticsIntegration(
        supabase_url=config['supabase_url'],
        supabase_key=config['supabase_key'],
        neural_config=config.get('neural_config', {}),
        device=config.get('device', 'cuda')
    )
