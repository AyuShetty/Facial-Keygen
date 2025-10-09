"""
BLOCKCHAIN INTEGRATION MODULE
============================

This module will integrate the facial key generation system with blockchain applications.
Designed to work with the research_facial_keygen_model.py

INTEGRATION ROADMAP:
1. Key Format Adaptation for different blockchain platforms
2. Smart Contract Integration 
3. Wallet Generation and Management
4. Multi-signature Implementation
5. Decentralized Identity System

SUPPORTED BLOCKCHAINS (Planned):
- Ethereum (ETH)
- Bitcoin (BTC) 
- Polygon (MATIC)
- Binance Smart Chain (BSC)
- Solana (SOL)

"""

import hashlib
import secrets
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import os
from datetime import datetime

# Import our research model
from research_facial_keygen_model import ResearchFacialKeygenModel

class BlockchainPlatform(Enum):
    ETHEREUM = "ethereum"
    BITCOIN = "bitcoin"
    POLYGON = "polygon"
    BSC = "binance_smart_chain"
    SOLANA = "solana"

@dataclass
class BlockchainKey:
    platform: BlockchainPlatform
    private_key: str
    public_key: str
    address: str
    derivation_path: Optional[str] = None
    facial_signature: Optional[str] = None

@dataclass
class WalletProfile:
    wallet_id: str
    facial_hash: str
    keys: Dict[BlockchainPlatform, BlockchainKey]
    created_at: datetime
    last_used: Optional[datetime] = None

class FacialBlockchainIntegrator:
    """
    Integrates facial key generation with blockchain platforms
    """
    
    def __init__(self, facial_model: ResearchFacialKeygenModel):
        self.facial_model = facial_model
        self.supported_platforms = list(BlockchainPlatform)
        
        # Platform-specific configurations
        self.platform_configs = {
            BlockchainPlatform.ETHEREUM: {
                'key_length': 32,
                'address_prefix': '0x',
                'derivation_path': "m/44'/60'/0'/0/0"
            },
            BlockchainPlatform.BITCOIN: {
                'key_length': 32,
                'address_prefix': '1',
                'derivation_path': "m/44'/0'/0'/0/0"
            },
            BlockchainPlatform.POLYGON: {
                'key_length': 32,
                'address_prefix': '0x',
                'derivation_path': "m/44'/60'/0'/0/0"
            },
            BlockchainPlatform.BSC: {
                'key_length': 32,
                'address_prefix': '0x',
                'derivation_path': "m/44'/60'/0'/0/0"
            },
            BlockchainPlatform.SOLANA: {
                'key_length': 32,
                'address_prefix': '',
                'derivation_path': "m/44'/501'/0'/0'"
            }
        }
        
        print("Facial Blockchain Integrator initialized")
        print(f"Supported platforms: {[p.value for p in self.supported_platforms]}")

    def generate_blockchain_keys_from_face(self, image_path: str, 
                                         platforms: List[BlockchainPlatform] = None) -> WalletProfile:
        """
        Generate blockchain keys from facial biometrics
        
        Args:
            image_path: Path to facial image
            platforms: List of blockchain platforms to generate keys for
            
        Returns:
            WalletProfile with keys for each platform
        """
        
        if platforms is None:
            platforms = [BlockchainPlatform.ETHEREUM]  # Default to Ethereum
        
        print(f"\n{'='*60}")
        print("GENERATING BLOCKCHAIN KEYS FROM FACIAL BIOMETRICS")
        print(f"{'='*60}")
        
        # Step 1: Generate facial keys using research model
        print("Step 1: Extracting facial biometric keys...")
        facial_keys = self.facial_model.research_pipeline(image_path)
        
        # Step 2: Create deterministic seed from facial data
        seed = self._create_deterministic_seed(facial_keys)
        facial_signature = hashlib.sha256(seed).hexdigest()
        
        print(f"Step 2: Generated deterministic seed")
        print(f"Facial signature: {facial_signature[:32]}...")
        
        # Step 3: Generate platform-specific keys
        print("Step 3: Generating platform-specific keys...")
        blockchain_keys = {}
        
        for platform in platforms:
            try:
                key = self._generate_platform_key(seed, platform)
                blockchain_keys[platform] = key
                print(f"  ✓ {platform.value}: {key.address}")
            except Exception as e:
                print(f"  ✗ {platform.value}: Failed - {e}")
        
        # Step 4: Create wallet profile
        wallet_profile = WalletProfile(
            wallet_id=f"facial_wallet_{facial_signature[:16]}",
            facial_hash=facial_signature,
            keys=blockchain_keys,
            created_at=datetime.now()
        )
        
        print(f"\nGenerated wallet profile: {wallet_profile.wallet_id}")
        print(f"{'='*60}")
        
        return wallet_profile

    def _create_deterministic_seed(self, facial_keys: Dict) -> bytes:
        """
        Create a deterministic seed from facial keys for blockchain generation
        """
        
        # Combine multiple facial key components for seed generation
        seed_components = [
            str(facial_keys['primary_numeric_key']),
            str(facial_keys['secondary_numeric_key']),
            str(facial_keys['blockchain_address']),
            facial_keys['sha256'],
            str(facial_keys['multi_signature_key'])
        ]
        
        # Create composite seed
        combined = ''.join(seed_components).encode('utf-8')
        
        # Use multiple hash rounds for additional security
        seed = combined
        for _ in range(10000):  # 10,000 rounds of hashing
            seed = hashlib.sha512(seed).digest()
        
        return seed[:32]  # Return 32 bytes for use as private key seed

    def _generate_platform_key(self, seed: bytes, platform: BlockchainPlatform) -> BlockchainKey:
        """
        Generate platform-specific blockchain key from seed
        """
        
        config = self.platform_configs[platform]
        
        # Create platform-specific seed
        platform_seed = hashlib.sha256(seed + platform.value.encode()).digest()
        
        # Generate private key (simplified - in production use proper key derivation)
        private_key = hashlib.sha256(platform_seed).hexdigest()
        
        # Generate public key (simplified - in production use elliptic curve cryptography)
        public_key = hashlib.sha256((private_key + "public").encode()).hexdigest()
        
        # Generate address based on platform
        address = self._generate_address(public_key, platform)
        
        return BlockchainKey(
            platform=platform,
            private_key=private_key,
            public_key=public_key,
            address=address,
            derivation_path=config.get('derivation_path'),
            facial_signature=hashlib.sha256(seed).hexdigest()[:32]
        )

    def _generate_address(self, public_key: str, platform: BlockchainPlatform) -> str:
        """
        Generate blockchain address from public key
        (Simplified implementation - production would use proper address generation)
        """
        
        config = self.platform_configs[platform]
        
        if platform in [BlockchainPlatform.ETHEREUM, BlockchainPlatform.POLYGON, BlockchainPlatform.BSC]:
            # Ethereum-style address
            address_hash = hashlib.keccak()  # Note: This would need proper Keccak implementation
            # Simplified: using SHA256 instead
            address_hash = hashlib.sha256(public_key.encode()).hexdigest()
            return config['address_prefix'] + address_hash[-40:]  # Last 20 bytes as hex
        
        elif platform == BlockchainPlatform.BITCOIN:
            # Bitcoin-style address (simplified)
            address_hash = hashlib.sha256(public_key.encode()).hexdigest()
            return config['address_prefix'] + address_hash[:34]  # Simplified format
        
        elif platform == BlockchainPlatform.SOLANA:
            # Solana-style address
            address_hash = hashlib.sha256(public_key.encode()).hexdigest()
            return address_hash[:44]  # Base58 format (simplified)
        
        else:
            raise ValueError(f"Unsupported platform: {platform}")

    def verify_facial_signature(self, image_path: str, wallet_profile: WalletProfile) -> bool:
        """
        Verify that an image matches the facial signature in wallet profile
        """
        
        print(f"\nVerifying facial signature for {wallet_profile.wallet_id}...")
        
        try:
            # Generate keys from the provided image
            facial_keys = self.facial_model.research_pipeline(image_path)
            seed = self._create_deterministic_seed(facial_keys)
            signature = hashlib.sha256(seed).hexdigest()
            
            # Compare with stored signature
            match = signature == wallet_profile.facial_hash
            
            print(f"Verification result: {'✓ MATCH' if match else '✗ NO MATCH'}")
            return match
            
        except Exception as e:
            print(f"Verification failed: {e}")
            return False

    def save_wallet_profile(self, wallet_profile: WalletProfile, output_dir: str = "wallets") -> str:
        """Save wallet profile to JSON file"""
        
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{wallet_profile.wallet_id}.json"
        filepath = os.path.join(output_dir, filename)
        
        # Convert to serializable format
        wallet_data = {
            'wallet_id': wallet_profile.wallet_id,
            'facial_hash': wallet_profile.facial_hash,
            'created_at': wallet_profile.created_at.isoformat(),
            'keys': {}
        }
        
        for platform, key in wallet_profile.keys.items():
            wallet_data['keys'][platform.value] = {
                'platform': key.platform.value,
                'private_key': key.private_key,
                'public_key': key.public_key,
                'address': key.address,
                'derivation_path': key.derivation_path,
                'facial_signature': key.facial_signature
            }
        
        with open(filepath, 'w') as f:
            json.dump(wallet_data, f, indent=2)
        
        print(f"Wallet profile saved: {filepath}")
        return filepath

    def load_wallet_profile(self, filepath: str) -> WalletProfile:
        """Load wallet profile from JSON file"""
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Reconstruct BlockchainKey objects
        keys = {}
        for platform_str, key_data in data['keys'].items():
            platform = BlockchainPlatform(platform_str)
            keys[platform] = BlockchainKey(
                platform=platform,
                private_key=key_data['private_key'],
                public_key=key_data['public_key'],
                address=key_data['address'],
                derivation_path=key_data.get('derivation_path'),
                facial_signature=key_data.get('facial_signature')
            )
        
        return WalletProfile(
            wallet_id=data['wallet_id'],
            facial_hash=data['facial_hash'],
            keys=keys,
            created_at=datetime.fromisoformat(data['created_at'])
        )

    def generate_multi_signature_setup(self, wallet_profiles: List[WalletProfile], 
                                     platform: BlockchainPlatform,
                                     threshold: int = 2) -> Dict:
        """
        Generate multi-signature wallet setup from multiple facial profiles
        """
        
        if len(wallet_profiles) < threshold:
            raise ValueError("Not enough wallet profiles for multi-signature setup")
        
        print(f"\nGenerating {threshold}-of-{len(wallet_profiles)} multi-signature setup for {platform.value}")
        
        # Extract addresses for the platform
        addresses = []
        for profile in wallet_profiles:
            if platform in profile.keys:
                addresses.append(profile.keys[platform].address)
        
        if len(addresses) < threshold:
            raise ValueError(f"Not enough {platform.value} addresses for multi-signature")
        
        # Create multi-sig configuration
        multisig_config = {
            'platform': platform.value,
            'type': f"{threshold}-of-{len(addresses)}",
            'threshold': threshold,
            'participants': addresses,
            'created_at': datetime.now().isoformat(),
            'facial_profiles': [p.wallet_id for p in wallet_profiles]
        }
        
        print(f"Multi-signature configuration created:")
        print(f"  Platform: {platform.value}")
        print(f"  Type: {threshold}-of-{len(addresses)}")
        print(f"  Participants: {len(addresses)}")
        
        return multisig_config

def demo_blockchain_integration():
    """
    Demonstration of blockchain integration with facial key generation
    """
    
    print("FACIAL BLOCKCHAIN INTEGRATION DEMO")
    print("=" * 60)
    
    # Initialize models
    print("Initializing facial keygen model...")
    facial_model = ResearchFacialKeygenModel(
        target_features=128,
        slot_count=16,
        lfsr_rounds=5,
        research_mode=False  # Reduce output for demo
    )
    
    # Check for images
    captures_dir = "captures"
    if not os.path.exists(captures_dir):
        print(f"Error: {captures_dir} directory not found!")
        return
    
    image_files = [f for f in os.listdir(captures_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        print(f"Error: No images found in {captures_dir}!")
        return
    
    image_paths = [os.path.join(captures_dir, f) for f in image_files[:3]]  # Use first 3 images
    
    # Train facial model
    print("Training facial model...")
    facial_model.train_research_model(image_paths)
    
    # Initialize blockchain integrator
    integrator = FacialBlockchainIntegrator(facial_model)
    
    # Generate blockchain wallets for each image
    platforms = [BlockchainPlatform.ETHEREUM, BlockchainPlatform.BITCOIN, BlockchainPlatform.POLYGON]
    wallet_profiles = []
    
    for i, image_path in enumerate(image_paths):
        print(f"\nProcessing image {i+1}: {os.path.basename(image_path)}")
        
        try:
            # Generate wallet
            wallet = integrator.generate_blockchain_keys_from_face(image_path, platforms)
            wallet_profiles.append(wallet)
            
            # Save wallet
            integrator.save_wallet_profile(wallet)
            
            # Display results
            print(f"\nWallet Generated: {wallet.wallet_id}")
            for platform, key in wallet.keys.items():
                print(f"  {platform.value}: {key.address}")
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    
    # Generate multi-signature setup
    if len(wallet_profiles) >= 2:
        print(f"\n{'='*60}")
        print("MULTI-SIGNATURE SETUP")
        print(f"{'='*60}")
        
        for platform in platforms:
            try:
                multisig = integrator.generate_multi_signature_setup(
                    wallet_profiles, platform, threshold=2
                )
                
                # Save multi-sig config
                multisig_file = f"multisig_{platform.value}.json"
                with open(multisig_file, 'w') as f:
                    json.dump(multisig, f, indent=2)
                print(f"Multi-sig config saved: {multisig_file}")
                
            except Exception as e:
                print(f"Error creating multi-sig for {platform.value}: {e}")
    
    print(f"\n{'='*60}")
    print("BLOCKCHAIN INTEGRATION DEMO COMPLETED")
    print(f"{'='*60}")
    print("Generated Files:")
    print("- Individual wallet profiles in 'wallets/' directory")
    print("- Multi-signature configurations")
    print("\nNext Steps:")
    print("1. Implement proper cryptographic key derivation")
    print("2. Add support for specific blockchain SDKs")
    print("3. Implement smart contract integration")
    print("4. Add transaction signing capabilities")

if __name__ == "__main__":
    demo_blockchain_integration()
