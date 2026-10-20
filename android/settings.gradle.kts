// Optional: set WALLMAP_MAVEN_MIRROR to a Maven Central mirror URL (e.g. when Central rate-limits you with 429),
// e.g. export WALLMAP_MAVEN_MIRROR=https://maven-central.storage-download.googleapis.com/maven2/
val mirror: String? = System.getenv("WALLMAP_MAVEN_MIRROR")

pluginManagement {
    val m: String? = System.getenv("WALLMAP_MAVEN_MIRROR")
    repositories {
        google()
        if (m != null) maven(url = m)
        mavenCentral()
        gradlePluginPortal()
    }
}
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        if (mirror != null) maven(url = mirror)
        mavenCentral()
    }
}
rootProject.name = "wallmap-logger"
include(":app")
