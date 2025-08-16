#!/usr/bin/env python3
"""
Deployment Validation Script for NIC ETL Pipeline
Validates deployment configuration and readiness
"""

import sys
import os
from pathlib import Path
import json
from datetime import datetime

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent.parent / "modules"))

def validate_deployment():
    """Validate deployment configuration and components"""
    print("🔍 NIC ETL Pipeline - Deployment Validation")
    print("=" * 50)
    
    validation_results = {
        "timestamp": datetime.utcnow().isoformat(),
        "checks": {},
        "overall_status": "unknown"
    }
    
    # 1. Check deployment files
    print("\n📁 Checking deployment files...")
    deployment_files = [
        "deployment/docker-compose.yml",
        "deployment/Dockerfile", 
        "deployment/start.sh",
        "deployment/health_check.py",
        "deployment/deploy.sh",
        "deployment/environments/.env.production",
        "deployment/environments/.env.staging",
        "requirements.txt"
    ]
    
    missing_files = []
    for file_path in deployment_files:
        full_path = Path(__file__).parent.parent / file_path
        if full_path.exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path}")
            missing_files.append(file_path)
    
    validation_results["checks"]["deployment_files"] = {
        "status": "pass" if not missing_files else "fail",
        "missing_files": missing_files,
        "total_files": len(deployment_files),
        "found_files": len(deployment_files) - len(missing_files)
    }
    
    # 2. Check module availability
    print("\n📦 Checking module availability...")
    modules = [
        'configuration_management',
        'error_handling',
        'metadata_management', 
        'gitlab_integration',
        'document_ingestion',
        'docling_processing',
        'text_chunking',
        'embedding_generation',
        'qdrant_integration',
        'pipeline_orchestration'
    ]
    
    available_modules = []
    unavailable_modules = []
    
    for module_name in modules:
        try:
            __import__(module_name)
            available_modules.append(module_name)
            print(f"  ✅ {module_name}")
        except ImportError as e:
            unavailable_modules.append(module_name)
            print(f"  ⚠️  {module_name}: {e}")
    
    module_availability = len(available_modules) / len(modules)
    validation_results["checks"]["modules"] = {
        "status": "pass" if module_availability >= 0.8 else "fail",
        "available": available_modules,
        "unavailable": unavailable_modules,
        "availability_rate": module_availability
    }
    
    # 3. Check configuration system
    print("\n⚙️  Checking configuration system...")
    try:
        from configuration_management import create_configuration_manager
        
        # Test different environments
        environments = ['development', 'staging', 'production']
        config_results = {}
        
        for env in environments:
            try:
                config_manager = create_configuration_manager(environment=env)
                config_results[env] = "success"
                print(f"  ✅ {env} environment")
            except Exception as e:
                config_results[env] = f"error: {e}"
                print(f"  ❌ {env} environment: {e}")
        
        validation_results["checks"]["configuration"] = {
            "status": "pass" if all("success" in result for result in config_results.values()) else "partial",
            "environments": config_results
        }
        
    except Exception as e:
        validation_results["checks"]["configuration"] = {
            "status": "fail",
            "error": str(e)
        }
        print(f"  ❌ Configuration system error: {e}")
    
    # 4. Check pipeline orchestration
    print("\n🎯 Checking pipeline orchestration...")
    try:
        from pipeline_orchestration import create_orchestrator_from_config_dict
        
        test_config = {
            'environment': 'development',
            'pipeline': {'max_concurrent_documents': 2}
        }
        
        orchestrator = create_orchestrator_from_config_dict(test_config)
        stats = orchestrator.get_orchestrator_statistics()
        
        validation_results["checks"]["orchestration"] = {
            "status": "pass",
            "orchestrator_created": True,
            "statistics": stats
        }
        print(f"  ✅ Pipeline orchestrator ready")
        
    except Exception as e:
        validation_results["checks"]["orchestration"] = {
            "status": "fail",
            "error": str(e)
        }
        print(f"  ❌ Pipeline orchestration error: {e}")
    
    # 5. Check Docker configuration
    print("\n🐳 Checking Docker configuration...")
    docker_compose_path = Path(__file__).parent / "docker-compose.yml"
    dockerfile_path = Path(__file__).parent / "Dockerfile"
    
    docker_checks = {
        "docker_compose": docker_compose_path.exists(),
        "dockerfile": dockerfile_path.exists(),
        "start_script": (Path(__file__).parent / "start.sh").exists()
    }
    
    for check, result in docker_checks.items():
        status = "✅" if result else "❌"
        print(f"  {status} {check}")
    
    validation_results["checks"]["docker"] = {
        "status": "pass" if all(docker_checks.values()) else "fail",
        "checks": docker_checks
    }
    
    # 6. Check monitoring configuration
    print("\n📊 Checking monitoring configuration...")
    monitoring_files = [
        Path(__file__).parent / "monitoring" / "prometheus.yml",
        Path(__file__).parent / "monitoring" / "grafana" / "datasources" / "prometheus.yml"
    ]
    
    monitoring_checks = {}
    for file_path in monitoring_files:
        monitoring_checks[file_path.name] = file_path.exists()
        status = "✅" if file_path.exists() else "❌"
        print(f"  {status} {file_path.name}")
    
    validation_results["checks"]["monitoring"] = {
        "status": "pass" if all(monitoring_checks.values()) else "partial",
        "files": monitoring_checks
    }
    
    # 7. Overall assessment
    print("\n📋 Overall Assessment...")
    
    check_statuses = [
        validation_results["checks"]["deployment_files"]["status"],
        validation_results["checks"]["modules"]["status"], 
        validation_results["checks"]["configuration"]["status"],
        validation_results["checks"]["orchestration"]["status"],
        validation_results["checks"]["docker"]["status"],
        validation_results["checks"]["monitoring"]["status"]
    ]
    
    passed_checks = sum(1 for status in check_statuses if status == "pass")
    total_checks = len(check_statuses)
    success_rate = passed_checks / total_checks
    
    if success_rate >= 0.9:
        overall_status = "ready"
        status_emoji = "🎉"
        status_message = "DEPLOYMENT READY"
    elif success_rate >= 0.7:
        overall_status = "partial"
        status_emoji = "⚠️"
        status_message = "DEPLOYMENT PARTIALLY READY"
    else:
        overall_status = "not_ready"
        status_emoji = "❌"
        status_message = "DEPLOYMENT NOT READY"
    
    validation_results["overall_status"] = overall_status
    validation_results["success_rate"] = success_rate
    validation_results["passed_checks"] = passed_checks
    validation_results["total_checks"] = total_checks
    
    print(f"\n{status_emoji} {status_message}")
    print(f"Success Rate: {success_rate:.1%} ({passed_checks}/{total_checks})")
    
    # Summary and recommendations
    print(f"\n📝 Summary:")
    print(f"  - Deployment files: {'✅' if validation_results['checks']['deployment_files']['status'] == 'pass' else '❌'}")
    print(f"  - Module availability: {'✅' if validation_results['checks']['modules']['status'] == 'pass' else '⚠️'} ({validation_results['checks']['modules']['availability_rate']:.1%})")
    print(f"  - Configuration system: {'✅' if validation_results['checks']['configuration']['status'] == 'pass' else '❌'}")
    print(f"  - Pipeline orchestration: {'✅' if validation_results['checks']['orchestration']['status'] == 'pass' else '❌'}")
    print(f"  - Docker configuration: {'✅' if validation_results['checks']['docker']['status'] == 'pass' else '❌'}")
    print(f"  - Monitoring setup: {'✅' if validation_results['checks']['monitoring']['status'] == 'pass' else '⚠️'}")
    
    if overall_status == "ready":
        print(f"\n🚀 Ready for deployment! Use:")
        print(f"   ./deployment/deploy.sh production")
    elif overall_status == "partial":
        print(f"\n🔧 Some issues need attention before deployment")
    else:
        print(f"\n🛠️  Significant issues must be resolved before deployment")
    
    # Save validation results
    results_file = Path(__file__).parent / f"validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(validation_results, f, indent=2)
    
    print(f"\n💾 Validation results saved to: {results_file}")
    
    return overall_status == "ready"

def main():
    """Main validation entry point"""
    try:
        deployment_ready = validate_deployment()
        return 0 if deployment_ready else 1
    except Exception as e:
        print(f"\n❌ Validation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)